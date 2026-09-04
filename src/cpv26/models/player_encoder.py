"""Role-aware player state encoder with a shared identity core."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any

from cpv26.graph.routes import PLAYER_ROLE_NAMES, normalize_endpoint_role

from ._torch import ModuleBase, nn, require_torch


class PlayerRole(str, Enum):
    BATTING = "batting"
    PITCHING = "pitching"
    DEFENSE = "defense"
    BASERUNNING = "baserunning"
    CATCHER = "catcher"


PLAYER_ROLES: tuple[str, ...] = PLAYER_ROLE_NAMES


def normalize_player_role(role: str | PlayerRole) -> str:
    """Return the canonical role name shared with graph route endpoints."""

    value = role.value if isinstance(role, PlayerRole) else str(role)
    normalized = normalize_endpoint_role("player", value)
    if normalized not in PLAYER_ROLES:
        allowed = ", ".join(PLAYER_ROLES)
        raise ValueError(f"unknown player role {value!r}; expected one of: {allowed}")
    return normalized


class _RoleAdapter(ModuleBase):
    def __init__(self, hidden_dim: int, feature_dim: int, dropout: float) -> None:
        require_torch()
        super().__init__()
        self.feature_dim = feature_dim
        self.delta = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, shared_embedding: Any, role_features: Any) -> Any:
        combined = torch_cat((shared_embedding, role_features), dim=-1)
        return self.norm(shared_embedding + self.gate(combined) * self.delta(combined))


def torch_cat(values: Sequence[Any], dim: int) -> Any:
    torch, _ = require_torch()
    return torch.cat(values, dim=dim)


class RoleAwarePlayerEncoder(ModuleBase):
    """Encode point-in-time player features into role-specific representations.

    The shared core contains identity-independent, common player state.  Each
    adapter sees its own batting, pitching, defense, baserunning, or catcher
    features and learns a gated residual relative to that core.
    """

    def __init__(
        self,
        shared_input_dim: int,
        role_feature_dims: Mapping[str | PlayerRole, int],
        *,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        active_roles: Iterable[str | PlayerRole] | None = None,
    ) -> None:
        require_torch()
        super().__init__()
        if shared_input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("shared_input_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        normalized_dims: dict[str, int] = {}
        for raw_role, value in role_feature_dims.items():
            role = normalize_player_role(raw_role)
            if role in normalized_dims:
                raise ValueError(f"duplicate role dimension after normalization: {role}")
            normalized_dims[role] = int(value)
        unknown = set(normalized_dims).difference(PLAYER_ROLES)
        if unknown:
            raise ValueError(f"unknown role dimensions: {sorted(unknown)}")
        if active_roles is None:
            normalized_active_roles = PLAYER_ROLES
        else:
            requested = tuple(normalize_player_role(role) for role in active_roles)
            if len(set(requested)) != len(requested):
                raise ValueError("active_roles contains duplicate roles")
            requested_set = set(requested)
            normalized_active_roles = tuple(
                role for role in PLAYER_ROLES if role in requested_set
            )
            inactive_dimensions = set(normalized_dims).difference(requested_set)
            if inactive_dimensions:
                raise ValueError(
                    "role_feature_dims contains inactive roles: "
                    f"{sorted(inactive_dimensions)}"
                )
        if not normalized_active_roles:
            raise ValueError("active_roles cannot be empty")
        for role in normalized_active_roles:
            normalized_dims.setdefault(role, 0)
        if any(value < 0 for value in normalized_dims.values()):
            raise ValueError("role feature dimensions must be non-negative")

        self.shared_input_dim = shared_input_dim
        self.hidden_dim = hidden_dim
        self.active_roles = normalized_active_roles
        self.role_feature_dims = normalized_dims
        self.shared_core = nn.Sequential(
            nn.Linear(shared_input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.adapters = nn.ModuleDict(
            {
                role: _RoleAdapter(hidden_dim, feature_dim, dropout)
                for role, feature_dim in normalized_dims.items()
            }
        )

    def encode_shared(self, shared_features: Any) -> Any:
        if shared_features.shape[-1] != self.shared_input_dim:
            raise ValueError(
                f"shared feature width is {shared_features.shape[-1]}; "
                f"expected {self.shared_input_dim}"
            )
        return self.shared_core(shared_features)

    def encode_role(
        self,
        shared_features: Any,
        role_features: Any | None,
        role: str | PlayerRole,
    ) -> Any:
        role_name = normalize_player_role(role)
        if role_name not in self.role_feature_dims:
            raise ValueError(f"player role {role_name!r} is not active in this encoder")
        expected_dim = self.role_feature_dims[role_name]
        if role_features is None:
            if expected_dim:
                raise ValueError(f"{role_name} features are required with width {expected_dim}")
            role_features = shared_features.new_empty((*shared_features.shape[:-1], 0))
        if role_features.shape[:-1] != shared_features.shape[:-1]:
            raise ValueError("shared and role feature batch dimensions must match")
        if role_features.shape[-1] != expected_dim:
            raise ValueError(
                f"{role_name} feature width is {role_features.shape[-1]}; expected {expected_dim}"
            )
        return self.adapters[role_name](self.encode_shared(shared_features), role_features)

    def encode_state(
        self,
        shared_features: Any,
        role_features: Mapping[str | PlayerRole, Any | None] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Encode the shared state and every active player-role channel.

        Missing role features are allowed only for adapters configured with
        width zero. This complete mapping is the contract consumed by the
        role-aware RelGNN backbone.
        """

        supplied: dict[str, Any | None] = {}
        for raw_role, features in (role_features or {}).items():
            role_name = normalize_player_role(raw_role)
            if role_name in supplied:
                raise ValueError(f"duplicate role features after normalization: {role_name}")
            supplied[role_name] = features
        inactive = set(supplied).difference(self.active_roles)
        if inactive:
            raise ValueError(f"player role features are not active: {sorted(inactive)}")

        shared_embedding = self.encode_shared(shared_features)
        encoded: dict[str, Any] = {}
        for role_name in self.active_roles:
            features = supplied.get(role_name)
            expected_dim = self.role_feature_dims[role_name]
            if features is None:
                if expected_dim:
                    raise ValueError(
                        f"{role_name} features are required with width {expected_dim}"
                    )
                features = shared_features.new_empty((*shared_features.shape[:-1], 0))
            if features.shape[:-1] != shared_features.shape[:-1]:
                raise ValueError("shared and role feature batch dimensions must match")
            if features.shape[-1] != expected_dim:
                raise ValueError(
                    f"{role_name} feature width is {features.shape[-1]}; "
                    f"expected {expected_dim}"
                )
            encoded[role_name] = self.adapters[role_name](shared_embedding, features)
        return shared_embedding, encoded

    def forward(
        self,
        shared_features: Any,
        role_features: Any,
        role: str | PlayerRole | None = None,
    ) -> Any:
        """Encode one role or a mapping of roles.

        For a single role, pass its tensor and ``role``.  To reuse the shared
        core for several roles, pass a mapping and omit ``role``.
        """

        if role is not None:
            return self.encode_role(shared_features, role_features, role)
        if not isinstance(role_features, Mapping):
            raise TypeError("role_features must be a mapping when role is omitted")
        shared_embedding = self.encode_shared(shared_features)
        encoded: dict[str, Any] = {}
        for raw_role, features in role_features.items():
            role_name = normalize_player_role(raw_role)
            if role_name not in self.role_feature_dims:
                raise ValueError(f"player role {role_name!r} is not active in this encoder")
            if role_name in encoded:
                raise ValueError(
                    f"duplicate role features after normalization: {role_name}"
                )
            expected_dim = self.role_feature_dims[role_name]
            if features is None:
                if expected_dim:
                    raise ValueError(f"{role_name} features are required with width {expected_dim}")
                features = shared_features.new_empty((*shared_features.shape[:-1], 0))
            if features.shape[:-1] != shared_features.shape[:-1]:
                raise ValueError("shared and role feature batch dimensions must match")
            if features.shape[-1] != expected_dim:
                raise ValueError(
                    f"{role_name} feature width is {features.shape[-1]}; expected {expected_dim}"
                )
            encoded[role_name] = self.adapters[role_name](shared_embedding, features)
        return encoded


PlayerEncoder = RoleAwarePlayerEncoder
