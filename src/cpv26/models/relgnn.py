"""Role-aware relational GNN with atomic-route attention."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

from cpv26.graph import (
    PLAYER_ROLE_NAMES,
    SHARED_PLAYER_ROLE,
    AtomicRoute,
    AtomicRouteBatch,
    GraphSnapshot,
    RouteRegistry,
    TorchAtomicRouteBatch,
    default_route_registry,
)

from ._torch import ModuleBase, nn, require_torch
from .player_encoder import PlayerRole, RoleAwarePlayerEncoder, normalize_player_role

_PLAYER_CHANNEL_PREFIX = "player__"


def _state_channel(node_type: str, role: str) -> str:
    if node_type == "player" and role != SHARED_PLAYER_ROLE:
        return f"{_PLAYER_CHANNEL_PREFIX}{role}"
    return node_type


def _gate_key(route_name: str, reverse: bool) -> str:
    direction = "reverse" if reverse else "forward"
    return f"{route_name}__{direction}"


@dataclass(frozen=True, slots=True)
class RelGNNState:
    """Entity states plus independently routed player-role states."""

    node_states: Mapping[str, Any]
    player_role_states: Mapping[str, Any]

    def __post_init__(self) -> None:
        normalized_roles: dict[str, Any] = {}
        for raw_role, state in self.player_role_states.items():
            role = normalize_player_role(raw_role)
            if role in normalized_roles:
                raise ValueError(f"duplicate player state after role normalization: {role}")
            normalized_roles[role] = state
        object.__setattr__(self, "node_states", MappingProxyType(dict(self.node_states)))
        object.__setattr__(
            self,
            "player_role_states",
            MappingProxyType(normalized_roles),
        )

    def channels(self) -> dict[str, Any]:
        """Return the internal channel mapping used by message passing."""

        channels = dict(self.node_states)
        channels.update(
            {
                f"{_PLAYER_CHANNEL_PREFIX}{role}": state
                for role, state in self.player_role_states.items()
            }
        )
        return channels


class _CompositeRouteAttention(ModuleBase):
    """Destination-query attention within one composite atomic route."""

    def __init__(
        self,
        hidden_dim: int,
        event_feature_dim: int,
        *,
        num_heads: int,
        dropout: float,
        include_publication_delay: bool,
    ) -> None:
        require_torch()
        super().__init__()
        self.hidden_dim = hidden_dim
        self.event_feature_dim = event_feature_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.include_publication_delay = include_publication_delay

        self.forward_source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reverse_source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.forward_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reverse_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.forward_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reverse_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.forward_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reverse_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.forward_output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reverse_output = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if event_feature_dim:
            self.event_encoder = nn.Sequential(
                nn.Linear(event_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        else:
            self.event_encoder = None
        temporal_dim = 3 if include_publication_delay else 2
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _temporal_context(self, batch: TorchAtomicRouteBatch) -> Any:
        torch, _ = require_torch()
        age_hours = batch.event_age_seconds / 3600.0
        signed_log_age = torch.sign(age_hours) * torch.log1p(torch.abs(age_hours))
        age_decay = torch.exp(-torch.abs(batch.event_age_seconds) / (86400.0 * 30.0))
        features = [signed_log_age, age_decay]
        if self.include_publication_delay:
            delay_hours = batch.publication_delay_seconds / 3600.0
            signed_log_delay = torch.sign(delay_hours) * torch.log1p(
                torch.abs(delay_hours)
            )
            features.append(signed_log_delay)
        return self.temporal_encoder(torch.stack(features, dim=-1))

    def aggregate(
        self,
        source_state: Any,
        destination_state: Any,
        batch: TorchAtomicRouteBatch,
        *,
        reverse: bool,
    ) -> tuple[Any, Any]:
        """Aggregate one direction with a softmax per destination and head."""

        torch, _ = require_torch()
        destination_count = int(destination_state.shape[0])
        if reverse:
            source_index = batch.destination_index
            destination_index = batch.source_index
            source_projection = self.reverse_source
            query_projection = self.reverse_query
            key_projection = self.reverse_key
            value_projection = self.reverse_value
            output_projection = self.reverse_output
        else:
            source_index = batch.source_index
            destination_index = batch.destination_index
            source_projection = self.forward_source
            query_projection = self.forward_query
            key_projection = self.forward_key
            value_projection = self.forward_value
            output_projection = self.forward_output

        if batch.num_edges == 0:
            empty_messages = destination_state.new_zeros(
                (destination_count, self.hidden_dim)
            )
            empty_mask = torch.zeros(
                destination_count,
                dtype=torch.bool,
                device=destination_state.device,
            )
            return empty_messages, empty_mask

        context = source_projection(source_state[source_index])
        if self.event_encoder is not None:
            context = context + self.event_encoder(batch.event_features)
        context = self.context_norm(context + self._temporal_context(batch))
        context = self.dropout(context)

        query = query_projection(destination_state[destination_index]).reshape(
            batch.num_edges,
            self.num_heads,
            self.head_dim,
        )
        key = key_projection(context).reshape(
            batch.num_edges,
            self.num_heads,
            self.head_dim,
        )
        value = value_projection(context).reshape(
            batch.num_edges,
            self.num_heads,
            self.head_dim,
        )
        scores = (query * key).sum(dim=-1) / math.sqrt(self.head_dim)

        positive_weight = batch.weights > 0
        weighted_scores = scores.masked_fill(~positive_weight.unsqueeze(-1), -torch.inf)
        destination_max = scores.new_full(
            (destination_count, self.num_heads),
            -torch.inf,
        )
        destination_max.scatter_reduce_(
            0,
            destination_index.unsqueeze(-1).expand(-1, self.num_heads),
            weighted_scores,
            reduce="amax",
            include_self=True,
        )
        gathered_max = destination_max[destination_index]
        centered_scores = torch.where(
            positive_weight.unsqueeze(-1),
            scores - gathered_max,
            torch.zeros_like(scores),
        )
        numerator = torch.exp(centered_scores) * batch.weights.unsqueeze(-1)
        denominator = scores.new_zeros((destination_count, self.num_heads))
        denominator.index_add_(0, destination_index, numerator)
        attention = numerator / denominator[destination_index].clamp_min(
            torch.finfo(scores.dtype).tiny
        )

        aggregated_heads = value.new_zeros(
            (destination_count, self.num_heads, self.head_dim)
        )
        aggregated_heads.index_add_(
            0,
            destination_index,
            attention.unsqueeze(-1) * value,
        )
        aggregate = output_projection(
            aggregated_heads.reshape(destination_count, self.hidden_dim)
        )
        route_mask = denominator.sum(dim=-1) > 0
        return aggregate, route_mask


class _CompositeRouteLayer(ModuleBase):
    """Combine route-local attention outputs with learned route gates."""

    def __init__(
        self,
        node_types: Iterable[str],
        route_feature_dims: Mapping[str, int],
        registry: RouteRegistry,
        *,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        include_publication_delay: bool,
    ) -> None:
        require_torch()
        super().__init__()
        self.hidden_dim = hidden_dim
        self.registry = registry
        self.messages = nn.ModuleDict(
            {
                route_name: _CompositeRouteAttention(
                    hidden_dim,
                    feature_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    include_publication_delay=include_publication_delay,
                )
                for route_name, feature_dim in route_feature_dims.items()
            }
        )
        self.route_gates = nn.ModuleDict()
        for route_name in route_feature_dims:
            route = registry.require(route_name)
            self.route_gates[_gate_key(route_name, False)] = self._new_route_gate()
            if route.bidirectional:
                self.route_gates[_gate_key(route_name, True)] = self._new_route_gate()

        channel_names = set(node_types)
        channel_names.update(
            f"{_PLAYER_CHANNEL_PREFIX}{role}" for role in PLAYER_ROLE_NAMES
        )
        self.updaters = nn.ModuleDict(
            {channel: nn.GRUCell(hidden_dim, hidden_dim) for channel in channel_names}
        )
        self.norms = nn.ModuleDict(
            {channel: nn.LayerNorm(hidden_dim) for channel in channel_names}
        )

    def _new_route_gate(self) -> Any:
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )

    @staticmethod
    def _endpoint_channel(
        route: AtomicRoute,
        *,
        source: bool,
        channels: Mapping[str, Any],
        strict_player_roles: bool,
    ) -> str:
        node_type = route.source_type if source else route.destination_type
        role = route.source_role if source else route.destination_role
        channel = _state_channel(node_type, role)
        if channel in channels:
            return channel
        if node_type == "player" and not strict_player_roles:
            return "player"
        endpoint = "source" if source else "destination"
        raise ValueError(
            f"route {route.name!r} requires {endpoint} player role state {role!r}"
        )

    @staticmethod
    def _validate_tensor_batch(
        route: AtomicRoute,
        batch: TorchAtomicRouteBatch,
        source_state: Any,
        destination_state: Any,
    ) -> None:
        torch, _ = require_torch()
        if (
            batch.source_type != route.source_type
            or batch.destination_type != route.destination_type
        ):
            raise ValueError(
                f"route {route.name!r} expects {route.source_type!r} -> "
                f"{route.destination_type!r}, got {batch.source_type!r} -> "
                f"{batch.destination_type!r}"
            )
        edge_count = batch.num_edges
        if batch.source_index.ndim != 1 or batch.destination_index.ndim != 1:
            raise ValueError("route endpoint indices must be one-dimensional")
        if int(batch.destination_index.numel()) != edge_count:
            raise ValueError("route source and destination indices must have equal length")
        temporal_columns = (
            batch.event_age_seconds,
            batch.publication_delay_seconds,
            batch.weights,
        )
        malformed_temporal = any(
            column.ndim != 1 or int(column.numel()) != edge_count
            for column in temporal_columns
        )
        if malformed_temporal:
            raise ValueError("route time and weight tensors must have one value per edge")
        if batch.event_features.ndim != 2 or int(batch.event_features.shape[0]) != edge_count:
            raise ValueError("route event features must have one row per edge")
        if not bool(torch.isfinite(batch.event_age_seconds).all().item()):
            raise ValueError("route event ages must be finite")
        if not bool(torch.isfinite(batch.publication_delay_seconds).all().item()):
            raise ValueError("route publication delays must be finite")
        if not bool(torch.isfinite(batch.weights).all().item()):
            raise ValueError("route weights must be finite")
        if bool((batch.weights < 0).any().item()):
            raise ValueError("route weights must be non-negative")
        availability_age = batch.event_age_seconds - batch.publication_delay_seconds
        if bool((availability_age < -1e-6).any().item()):
            raise ValueError(
                f"route {route.name!r} contains information available after its cutoff"
            )
        if bool((batch.source_index < 0).any().item()) or bool(
            (batch.destination_index < 0).any().item()
        ):
            raise IndexError("route indices must be non-negative")
        if batch.num_edges:
            if int(batch.source_index.max().item()) >= int(source_state.shape[0]):
                raise IndexError(f"source index exceeds {route.source_type!r} node count")
            if int(batch.destination_index.max().item()) >= int(
                destination_state.shape[0]
            ):
                raise IndexError(
                    f"destination index exceeds {route.destination_type!r} node count"
                )

    def forward(
        self,
        state: RelGNNState,
        route_batches: tuple[TorchAtomicRouteBatch, ...],
    ) -> RelGNNState:
        torch, _ = require_torch()
        channels = state.channels()
        strict_player_roles = bool(state.player_role_states)
        incoming: dict[str, list[tuple[Any, Any, str]]] = {
            channel: [] for channel in channels
        }

        for batch in route_batches:
            route = self.registry.require(batch.route_name)
            if batch.route_name not in self.messages:
                raise ValueError(f"route {batch.route_name!r} is not enabled in this backbone")
            source_channel = self._endpoint_channel(
                route,
                source=True,
                channels=channels,
                strict_player_roles=strict_player_roles,
            )
            destination_channel = self._endpoint_channel(
                route,
                source=False,
                channels=channels,
                strict_player_roles=strict_player_roles,
            )
            source_state = channels[source_channel]
            destination_state = channels[destination_channel]
            self._validate_tensor_batch(route, batch, source_state, destination_state)

            message, mask = self.messages[batch.route_name].aggregate(
                source_state,
                destination_state,
                batch,
                reverse=False,
            )
            incoming[destination_channel].append(
                (message, mask, _gate_key(batch.route_name, False))
            )
            if route.bidirectional:
                reverse_message, reverse_mask = self.messages[batch.route_name].aggregate(
                    destination_state,
                    source_state,
                    batch,
                    reverse=True,
                )
                incoming[source_channel].append(
                    (reverse_message, reverse_mask, _gate_key(batch.route_name, True))
                )

        updated_channels = dict(channels)
        for channel, route_messages in incoming.items():
            previous = channels[channel]
            if not route_messages or int(previous.shape[0]) == 0:
                continue
            messages = torch.stack([item[0] for item in route_messages], dim=1)
            masks = torch.stack([item[1] for item in route_messages], dim=1)
            gate_scores = torch.cat(
                [
                    self.route_gates[gate_key](torch.cat((previous, message), dim=-1))
                    for message, _, gate_key in route_messages
                ],
                dim=1,
            )
            gate_scores = gate_scores.masked_fill(~masks, -torch.inf)
            any_message = masks.any(dim=1)
            safe_scores = torch.where(
                any_message.unsqueeze(-1),
                gate_scores,
                torch.zeros_like(gate_scores),
            )
            route_attention = torch.softmax(safe_scores, dim=1) * masks
            route_attention = route_attention / route_attention.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(torch.finfo(route_attention.dtype).tiny)
            combined = (route_attention.unsqueeze(-1) * messages).sum(dim=1)
            candidate = self.norms[channel](self.updaters[channel](combined, previous))
            updated_channels[channel] = torch.where(
                any_message.unsqueeze(-1),
                candidate,
                previous,
            )

        node_states = {
            node_type: updated_channels[node_type] for node_type in state.node_states
        }
        player_role_states = {
            role: updated_channels[f"{_PLAYER_CHANNEL_PREFIX}{role}"]
            for role in state.player_role_states
        }
        return RelGNNState(node_states, player_role_states)


class CompositeRelGNNBackbone(ModuleBase):
    """RelGNN backbone with two-level attention over reviewed atomic routes.

    Each route first performs destination-query multi-head attention over its
    observed events. A learned route gate then combines the independent route
    summaries for each destination node. When ``player_encoder`` or explicit
    ``player_role_states`` are supplied, player endpoints select the canonical
    batting, pitching, defense, baserunning, or catcher channel declared by the
    route. Calls without role states retain the legacy shared-player behavior.
    """

    def __init__(
        self,
        node_input_dims: Mapping[str, int] | None = None,
        route_feature_dims: Mapping[str, int] | None = None,
        *,
        node_feature_dims: Mapping[str, int] | None = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        include_publication_delay: bool = False,
        player_encoder: RoleAwarePlayerEncoder | None = None,
        registry: RouteRegistry | None = None,
    ) -> None:
        require_torch()
        super().__init__()
        if node_input_dims is not None and node_feature_dims is not None:
            raise ValueError("pass either node_input_dims or node_feature_dims, not both")
        configured_node_dims = node_input_dims or node_feature_dims
        if not configured_node_dims:
            raise ValueError("node_input_dims cannot be empty")
        if not route_feature_dims:
            raise ValueError("route_feature_dims cannot be empty")
        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be positive")
        if num_attention_heads <= 0 or hidden_dim % num_attention_heads:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        normalized_node_dims = {
            node_type: int(feature_dim)
            for node_type, feature_dim in configured_node_dims.items()
        }
        normalized_route_dims = {
            route_name: int(feature_dim)
            for route_name, feature_dim in route_feature_dims.items()
        }
        if any(feature_dim <= 0 for feature_dim in normalized_node_dims.values()):
            raise ValueError("node feature dimensions must be positive")
        if any(feature_dim < 0 for feature_dim in normalized_route_dims.values()):
            raise ValueError("route feature dimensions must be non-negative")

        self.registry = (registry or default_route_registry()).copy()
        for route_name in normalized_route_dims:
            route = self.registry.require(route_name)
            if route.source_type not in normalized_node_dims:
                raise ValueError(
                    f"route {route_name!r} source type {route.source_type!r} "
                    "has no node encoder"
                )
            if route.destination_type not in normalized_node_dims:
                raise ValueError(
                    f"route {route_name!r} destination type "
                    f"{route.destination_type!r} has no node encoder"
                )

        if player_encoder is not None:
            if "player" not in normalized_node_dims:
                raise ValueError("player_encoder requires a 'player' node type")
            if player_encoder.shared_input_dim != normalized_node_dims["player"]:
                raise ValueError(
                    "player_encoder shared_input_dim must match the player node feature width"
                )
            if player_encoder.hidden_dim != hidden_dim:
                raise ValueError("player_encoder hidden_dim must match backbone hidden_dim")

        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.include_publication_delay = include_publication_delay
        self.node_feature_dims = normalized_node_dims
        self.node_input_dims = normalized_node_dims
        self.route_feature_dims = normalized_route_dims
        self.player_encoder = player_encoder
        self.node_encoders = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                for node_type, feature_dim in normalized_node_dims.items()
                if node_type != "player" or player_encoder is None
            }
        )
        self.layers = nn.ModuleList(
            [
                _CompositeRouteLayer(
                    normalized_node_dims,
                    normalized_route_dims,
                    self.registry,
                    hidden_dim=hidden_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    include_publication_delay=include_publication_delay,
                )
                for _ in range(num_layers)
            ]
        )

    def _tensorize_routes(
        self,
        route_batches: Iterable[AtomicRouteBatch | TorchAtomicRouteBatch],
        *,
        cutoff_at: datetime | None,
        device: Any,
        dtype: Any,
    ) -> tuple[TorchAtomicRouteBatch, ...]:
        tensor_batches: list[TorchAtomicRouteBatch] = []
        for batch in route_batches:
            if isinstance(batch, TorchAtomicRouteBatch):
                tensor_batch = batch
            elif isinstance(batch, AtomicRouteBatch):
                if cutoff_at is None:
                    raise ValueError("cutoff_at is required for non-tensor route batches")
                tensor_batch = batch.as_torch(
                    cutoff_at=cutoff_at,
                    registry=self.registry,
                    device=device,
                    dtype=dtype,
                )
            else:
                raise TypeError(f"unsupported route batch type: {type(batch).__name__}")
            expected_dim = self.route_feature_dims.get(tensor_batch.route_name)
            if expected_dim is None:
                raise ValueError(
                    f"route {tensor_batch.route_name!r} is not enabled in this backbone"
                )
            if int(tensor_batch.event_features.shape[-1]) != expected_dim:
                raise ValueError(
                    f"route {tensor_batch.route_name!r} event feature width is "
                    f"{tensor_batch.event_features.shape[-1]}; expected {expected_dim}"
                )
            tensor_batches.append(tensor_batch)
        return tuple(tensor_batches)

    def _required_player_roles(self) -> set[str]:
        required: set[str] = set()
        for route_name in self.route_feature_dims:
            route = self.registry.require(route_name)
            for node_type, role in (
                (route.source_type, route.source_role),
                (route.destination_type, route.destination_role),
            ):
                if node_type == "player" and role != SHARED_PLAYER_ROLE:
                    required.add(role)
        return required

    def _encode_initial_state(
        self,
        node_features: Mapping[str, Any],
        *,
        player_role_features: Mapping[str | PlayerRole, Any | None] | None,
        player_role_states: Mapping[str | PlayerRole, Any] | None,
    ) -> RelGNNState:
        missing = set(self.node_feature_dims).difference(node_features)
        if missing:
            raise ValueError(f"missing node feature matrices: {sorted(missing)}")
        extra = set(node_features).difference(self.node_feature_dims)
        if extra:
            raise ValueError(f"unknown node feature matrices: {sorted(extra)}")
        for node_type, expected_dim in self.node_feature_dims.items():
            features = node_features[node_type]
            if features.ndim != 2 or int(features.shape[-1]) != expected_dim:
                raise ValueError(
                    f"{node_type!r} node feature matrix must have shape [N, {expected_dim}]"
                )

        encoded_nodes = {
            node_type: self.node_encoders[node_type](features)
            for node_type, features in node_features.items()
            if node_type != "player" or self.player_encoder is None
        }
        encoded_roles: dict[str, Any] = {}
        if self.player_encoder is not None:
            if player_role_states is not None:
                raise ValueError(
                    "player_role_states cannot be combined with a configured player_encoder"
                )
            shared_player, encoded_roles = self.player_encoder.encode_state(
                node_features["player"],
                player_role_features,
            )
            encoded_nodes["player"] = shared_player
        else:
            if player_role_features is not None:
                raise ValueError(
                    "player_role_features require a configured RoleAwarePlayerEncoder"
                )
            for raw_role, role_state in (player_role_states or {}).items():
                role = normalize_player_role(raw_role)
                if role in encoded_roles:
                    raise ValueError(
                        f"duplicate player state after role normalization: {role}"
                    )
                if role_state.ndim != 2 or int(role_state.shape[-1]) != self.hidden_dim:
                    raise ValueError(
                        f"player role state {role!r} must have shape [N, {self.hidden_dim}]"
                    )
                if int(role_state.shape[0]) != int(encoded_nodes["player"].shape[0]):
                    raise ValueError(
                        f"player role state {role!r} row count must match player nodes"
                    )
                encoded_roles[role] = role_state

        if self.player_encoder is not None or player_role_states is not None:
            missing_roles = self._required_player_roles().difference(encoded_roles)
            if missing_roles:
                raise ValueError(
                    "missing player role states required by enabled routes: "
                    f"{sorted(missing_roles)}"
                )
        return RelGNNState(encoded_nodes, encoded_roles)

    def forward_relational_state(
        self,
        node_features: Mapping[str, Any],
        route_batches: Iterable[AtomicRouteBatch | TorchAtomicRouteBatch],
        *,
        cutoff_at: datetime | None = None,
        player_role_features: Mapping[str | PlayerRole, Any | None] | None = None,
        player_role_states: Mapping[str | PlayerRole, Any] | None = None,
    ) -> RelGNNState:
        """Return entity and player-role states after relational propagation."""

        first_features = next(iter(node_features.values()), None)
        if first_features is None:
            raise ValueError("node_features must not be empty")
        state = self._encode_initial_state(
            node_features,
            player_role_features=player_role_features,
            player_role_states=player_role_states,
        )
        tensor_batches = self._tensorize_routes(
            route_batches,
            cutoff_at=cutoff_at,
            device=first_features.device,
            dtype=first_features.dtype,
        )
        for layer in self.layers:
            state = layer(state, tensor_batches)
        return state

    def forward(
        self,
        node_features: Mapping[str, Any],
        route_batches: Iterable[AtomicRouteBatch | TorchAtomicRouteBatch],
        *,
        cutoff_at: datetime | None = None,
        player_role_features: Mapping[str | PlayerRole, Any | None] | None = None,
        player_role_states: Mapping[str | PlayerRole, Any] | None = None,
    ) -> dict[str, Any]:
        """Return entity states, preserving the original backbone return type."""

        state = self.forward_relational_state(
            node_features,
            route_batches,
            cutoff_at=cutoff_at,
            player_role_features=player_role_features,
            player_role_states=player_role_states,
        )
        return dict(state.node_states)

    def forward_snapshot(
        self,
        snapshot: GraphSnapshot,
        *,
        device: Any = None,
        dtype: Any = None,
        player_role_features: Mapping[str | PlayerRole, Any | None] | None = None,
        player_role_states: Mapping[str | PlayerRole, Any] | None = None,
    ) -> dict[str, Any]:
        """Tensorize and encode a validated point-in-time graph snapshot."""

        torch, _ = require_torch()
        parameter = next(self.parameters())
        target_device = device if device is not None else parameter.device
        target_dtype = dtype if dtype is not None else parameter.dtype
        node_features = snapshot.torch_node_features(
            device=target_device,
            dtype=target_dtype,
        )
        with torch.set_grad_enabled(self.training):
            return self.forward(
                node_features,
                snapshot.routes,
                cutoff_at=snapshot.cutoff_at,
                player_role_features=player_role_features,
                player_role_states=player_role_states,
            )


RelGNNBackbone = CompositeRelGNNBackbone
