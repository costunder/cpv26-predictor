"""Actual KBO RelGNN multitask model and disjoint daily-graph minibatches.

Only historical relations enter the shared graph. Match queries consume team
states, never today's observed batting order or first pitcher. Live Hit learns
a conditional-on-appearance joint PA/hit distribution; absent roster entries
are not invented as negative appearance labels.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from cpv26.graph import AtomicRoute, RouteRegistry, TorchAtomicRouteBatch

from ._torch import ModuleBase, nn, require_torch
from .heads import DirectRunDistributionHead, WDLHead
from .interaction import PlateAppearanceInteractionDecoder
from .player_encoder import RoleAwarePlayerEncoder
from .relgnn import CompositeRelGNNBackbone

KBO_ROUTE_NAMES = (
    "batter_pa_pitcher",
    "batter_participation_team",
    "pitcher_participation_team",
    "home_team_game_away_team",
)


def kbo_route_registry() -> RouteRegistry:
    """Reviewed historical participation routes, not roster/lineup aliases."""

    return RouteRegistry(
        (
            AtomicRoute(
                "batter_pa_pitcher",
                "player",
                "observed_plate_appearance",
                "player",
                "batting",
                "pitching",
            ),
            AtomicRoute(
                "batter_participation_team",
                "player",
                "observed_batting_participation",
                "team",
                "batting",
                "batting_team",
            ),
            AtomicRoute(
                "pitcher_participation_team",
                "player",
                "observed_pitching_participation",
                "team",
                "pitching",
                "fielding_team",
            ),
            AtomicRoute(
                "home_team_game_away_team",
                "team",
                "completed_game",
                "team",
                "home_team",
                "away_team",
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class KBORelGNNConfig:
    node_feature_dims: Mapping[str, int]
    role_feature_dims: Mapping[str, int]
    route_feature_dims: Mapping[str, int]
    hidden_dim: int = 64
    num_layers: int = 2
    num_attention_heads: int = 4
    dropout: float = 0.1
    max_plate_appearances: int = 8
    max_hits: int = 5
    pa_context_dim: int = 10
    include_run_head: bool = True
    include_boxscore_heads: bool = False
    box_batting_feature_dim: int = 19
    box_pitching_feature_dim: int = 21
    box_gradient_mode: str = "shared"

    def __post_init__(self) -> None:
        for name in ("node_feature_dims", "role_feature_dims", "route_feature_dims"):
            values = dict(getattr(self, name))
            if any(
                isinstance(value, bool) or not isinstance(value, int) for value in values.values()
            ):
                raise ValueError(f"{name} must contain integer dimensions")
            object.__setattr__(self, name, values)
        if set(self.node_feature_dims) != {"player", "team"}:
            raise ValueError("KBO graph requires exactly player and team node features")
        if any(width < 1 for width in self.node_feature_dims.values()):
            raise ValueError("node feature widths must be positive")
        if set(self.role_feature_dims) - {"batting", "pitching"}:
            raise ValueError("KBO role features must be batting/pitching observations")
        if any(width < 0 for width in self.role_feature_dims.values()):
            raise ValueError("role feature widths cannot be negative")
        if not self.route_feature_dims or set(self.route_feature_dims) - set(KBO_ROUTE_NAMES):
            raise ValueError("route_feature_dims must name reviewed KBO routes")
        if any(width < 0 for width in self.route_feature_dims.values()):
            raise ValueError("route feature widths cannot be negative")
        if self.hidden_dim < 1 or self.num_layers < 1 or self.num_attention_heads < 1:
            raise ValueError("model dimensions and layer/head counts must be positive")
        if self.hidden_dim % self.num_attention_heads:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if not 1 <= self.max_hits <= self.max_plate_appearances:
            raise ValueError("count supports require 1 <= max_hits <= max_plate_appearances")
        if self.pa_context_dim < 0:
            raise ValueError("pa_context_dim cannot be negative")
        if not isinstance(self.include_boxscore_heads, bool):
            raise ValueError("include_boxscore_heads must be boolean")
        if self.box_batting_feature_dim < 1 or self.box_pitching_feature_dim < 1:
            raise ValueError("box-score feature widths must be positive")
        if self.box_gradient_mode not in {"shared", "head_only"}:
            raise ValueError("box_gradient_mode must be shared or head_only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_feature_dims": dict(self.node_feature_dims),
            "role_feature_dims": dict(self.role_feature_dims),
            "route_feature_dims": dict(self.route_feature_dims),
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "dropout": self.dropout,
            "max_plate_appearances": self.max_plate_appearances,
            "max_hits": self.max_hits,
            "pa_context_dim": self.pa_context_dim,
            "include_run_head": self.include_run_head,
            "include_boxscore_heads": self.include_boxscore_heads,
            "box_batting_feature_dim": self.box_batting_feature_dim,
            "box_pitching_feature_dim": self.box_pitching_feature_dim,
            "box_gradient_mode": self.box_gradient_mode,
        }


class KBORelGNNModel(ModuleBase):
    """Role-aware RelGNN with separate match, conditional Live Hit, and PA heads."""

    def __init__(self, config: KBORelGNNConfig) -> None:
        torch, _ = require_torch()
        super().__init__()
        self.config = config
        node_widths = dict(config.node_feature_dims)
        role_widths = dict(config.role_feature_dims)
        if config.include_boxscore_heads:
            for kind in node_widths:
                node_widths[kind] += (
                    config.box_batting_feature_dim + config.box_pitching_feature_dim
                )
            for role, width in (
                ("batting", config.box_batting_feature_dim),
                ("pitching", config.box_pitching_feature_dim),
            ):
                role_widths[role] = role_widths.get(role, 0) + width
        encoder = RoleAwarePlayerEncoder(
            node_widths["player"],
            role_widths,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.backbone: Any = CompositeRelGNNBackbone(
            node_feature_dims=node_widths,
            route_feature_dims=config.route_feature_dims,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            dropout=config.dropout,
            player_encoder=encoder,
            registry=kbo_route_registry(),
        )
        self.match_head: Any = WDLHead(
            config.hidden_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        self.live_hit_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 5, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.hidden_dim * 2, (config.max_plate_appearances + 1) * (config.max_hits + 2)
            ),
        )
        self.pa_head: Any = PlateAppearanceInteractionDecoder(
            config.hidden_dim,
            context_dim=config.pa_context_dim,
            hidden_dim=config.hidden_dim * 2,
            dropout=config.dropout,
        )
        self.run_head: Any = (
            DirectRunDistributionHead(
                config.hidden_dim,
                hidden_dim=config.hidden_dim * 2,
                dropout=config.dropout,
            )
            if config.include_run_head
            else None
        )
        # Separate aggregate decoders never invent a particular opposing pitcher
        # or a current-PA base/out state from a historical box score.
        self.box_pa_head: Any = None
        self.box_pitch_head: Any = None
        if config.include_boxscore_heads:
            self.box_pa_head = nn.Sequential(
                nn.Linear(config.hidden_dim * 5, config.hidden_dim * 2),
                nn.LayerNorm(config.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim * 2, 10),
            )
            self.box_pitch_head = nn.Sequential(
                nn.Linear(config.hidden_dim * 5, config.hidden_dim * 2),
                nn.LayerNorm(config.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim * 2, 10),
            )
        pa_support = torch.arange(1, config.max_plate_appearances + 2, dtype=torch.float32)
        hit_support = torch.arange(config.max_hits + 2, dtype=torch.float32)
        # Final supports are overflow-bucket lower bounds (9+ PA, 6+ H by default).
        allowed = hit_support.unsqueeze(0) <= pa_support.unsqueeze(1)
        cast(Any, self).register_buffer("joint_allowed", allowed)
        cast(Any, self).register_buffer("pa_support", pa_support)
        cast(Any, self).register_buffer("hit_support", hit_support)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        torch, _ = require_torch()
        if batch.get("_validated_on_cpu") is not True:
            raise ValueError("use collate_kbo_day_graphs to validate graph indices before forward")
        nodes = batch["node_features"]
        roles = batch["role_features"]
        if self.config.include_boxscore_heads:
            nodes = {
                kind: torch.cat(
                    (
                        values,
                        batch[f"{kind}_box_batting_features"],
                        batch[f"{kind}_box_pitching_features"],
                    ),
                    dim=-1,
                )
                for kind, values in nodes.items()
            }
            roles = {
                role: torch.cat((roles[role], batch[f"player_box_{role}_features"]), dim=-1)
                for role in ("batting", "pitching")
            }
        elif any(batch[f"{task}_player_index"].numel() for task in ("box_pa", "box_pitch")):
            raise ValueError("historical box-score queries require include_boxscore_heads")
        state = self.backbone.forward_relational_state(
            nodes,
            batch["routes"],
            player_role_features=roles,
            validate_routes=False,
        )
        teams = state.node_states["team"]
        batting = state.player_role_states["batting"]
        pitching = state.player_role_states["pitching"]
        home = teams[batch["match_home_team_index"]]
        away = teams[batch["match_away_team_index"]]
        match_logits = self.match_head(home, away)
        player = batting[batch["live_hit_player_index"]]
        offense = teams[batch["live_hit_team_index"]]
        defense = teams[batch["live_hit_opponent_index"]]
        live_features = torch.cat(
            (player, offense, defense, player * defense, offense - defense), dim=-1
        )
        joint_logits = (
            self.live_hit_head(live_features)
            .reshape(
                -1,
                self.config.max_plate_appearances + 1,
                self.config.max_hits + 2,
            )
            .float()
        )
        joint_logits = joint_logits.masked_fill(~cast(Any, self).joint_allowed, -torch.inf)
        joint = torch.softmax(joint_logits.flatten(1), dim=-1).reshape(joint_logits.shape)
        output: dict[str, Any] = {
            "match_logits": match_logits,
            "live_hit_joint_logits": joint_logits,
            "live_hit_joint_probabilities": joint,
            "live_hit_hit_probability": joint[:, :, 1:].sum(dim=(1, 2)),
            "live_hit_expected_hits": (joint * cast(Any, self).hit_support).sum(dim=(1, 2)),
            "live_hit_expected_pa": (joint * cast(Any, self).pa_support[:, None]).sum(dim=(1, 2)),
            "pa_logits": self.pa_head(
                batting[batch["pa_batter_index"]],
                pitching[batch["pa_pitcher_index"]],
                game_context=batch["pa_context"],
            ),
        }
        if self.run_head is not None:
            output["match_run_parameters"] = self.run_head(home, away)
        if self.config.include_boxscore_heads:
            for task, player_states, head in (
                ("box_pa", batting, self.box_pa_head),
                ("box_pitch", pitching, self.box_pitch_head),
            ):
                player = player_states[batch[f"{task}_player_index"]]
                team = teams[batch[f"{task}_team_index"]]
                opponent = teams[batch[f"{task}_opponent_index"]]
                features = torch.cat(
                    (player, team, opponent, player * opponent, team - opponent), dim=-1
                )
                if self.config.box_gradient_mode == "head_only":
                    # This opt-in policy trains the aggregate heads but prevents
                    # their labels from changing the shared representation.
                    features = features.detach()
                values = head(features).float()
                if task == "box_pa":
                    output["box_pa_logits"] = values
                else:
                    output["box_pitch_rates"] = torch.nn.functional.softplus(values) + 1e-6
        return output


def encode_live_hit_targets(
    plate_appearances: Any,
    hits: Any,
    config: KBORelGNNConfig,
    *,
    validate: bool = True,
) -> Any:
    """Return flat masked-joint indices, with explicit positive-PA overflow bins."""

    torch, _ = require_torch()
    if plate_appearances.shape != hits.shape or plate_appearances.ndim != 1:
        raise ValueError("PA/hit targets must be matching one-dimensional tensors")
    if plate_appearances.is_floating_point() or hits.is_floating_point():
        raise TypeError("PA/hit targets must be integer tensors")
    if plate_appearances.dtype == torch.bool or hits.dtype == torch.bool:
        raise TypeError("PA/hit targets cannot be boolean")
    if validate and bool(
        ((plate_appearances < 1) | (hits < 0) | (hits > plate_appearances)).any().item()
    ):
        raise ValueError("conditional Live Hit targets require PA >= 1 and 0 <= H <= PA")
    pa_bucket = plate_appearances.long().clamp_max(config.max_plate_appearances + 1) - 1
    hit_bucket = hits.long().clamp_max(config.max_hits + 1)
    return pa_bucket * (config.max_hits + 2) + hit_bucket


def kbo_multitask_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    match_weight: float = 1.0,
    live_hit_weight: float = 1.0,
    pa_weight: float = 0.2,
    run_weight: float = 0.0,
    box_pa_weight: float = 0.2,
    box_pitch_weight: float = 0.1,
) -> dict[str, Any]:
    """Distinct task means plus their weighted sum; NB2 run NLL is optional."""

    torch, _ = require_torch()
    weights = (
        match_weight, live_hit_weight, pa_weight, run_weight, box_pa_weight, box_pitch_weight
    )
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("loss weights must be finite and non-negative")
    if batch.get("_validated_on_cpu") is not True:
        raise ValueError("loss requires a validated KBO graph batch")
    zero = outputs["match_logits"].sum() * 0.0
    match = (
        torch.nn.functional.cross_entropy(outputs["match_logits"].float(), batch["match_targets"])
        if batch["match_targets"].numel()
        else zero
    )
    joint_logits = outputs["live_hit_joint_logits"]
    if batch["live_hit_pa"].numel():
        live = live_hit_observed_nll(joint_logits, batch).mean()
    else:
        live = zero
    pa = (
        torch.nn.functional.cross_entropy(outputs["pa_logits"].float(), batch["pa_targets"])
        if batch["pa_targets"].numel()
        else zero
    )
    run = zero
    if "match_run_parameters" in outputs and batch["match_targets"].numel():
        parameters = outputs["match_run_parameters"]
        scores = batch["match_runs"].float()
        run = -(
            DirectRunDistributionHead.negative_binomial_log_prob(
                scores[:, 0],
                parameters["home_mean"].float(),
                parameters["home_dispersion"].float(),
            )
            + DirectRunDistributionHead.negative_binomial_log_prob(
                scores[:, 1],
                parameters["away_mean"].float(),
                parameters["away_dispersion"].float(),
            )
        ).mean()
    result = {
        "loss": match_weight * match + live_hit_weight * live + pa_weight * pa + run_weight * run,
        "match_loss": match,
        "live_hit_loss": live,
        "pa_loss": pa,
        "run_loss": run,
    }
    if "box_pa_logits" in outputs:
        counts = batch["box_pa_counts"].float()
        box_pa = zero
        if counts.numel():
            log_probabilities = torch.log_softmax(outputs["box_pa_logits"].float(), dim=-1)
            box_pa = -(counts * log_probabilities).sum() / counts.sum().clamp_min(1)
        mask = batch["box_pitch_mask"]
        box_pitch = zero
        if mask.numel():
            rates = outputs["box_pitch_rates"].float()
            targets = batch["box_pitch_targets"].float()
            # Full Poisson NLL; unobserved cells have no likelihood contribution.
            nll = rates - targets * rates.log() + torch.lgamma(targets + 1)
            box_pitch = nll.masked_fill(~mask, 0).sum() / mask.sum().clamp_min(1)
        result.update(box_pa_loss=box_pa, box_pitch_loss=box_pitch)
        result["loss"] = result["loss"] + box_pa_weight * box_pa + box_pitch_weight * box_pitch
    return result


def live_hit_observed_nll(joint_logits: Any, batch: Mapping[str, Any]) -> Any:
    """Score exact PA/H or the observed event {PA >= minimum, H = observed}.

    Unknown PA is -1, never a fabricated count. Counts above the final support
    use the existing overflow bucket; that bucket cannot resolve its interior.
    """
    torch, _ = require_torch()
    logits = joint_logits.float()
    pa = batch["live_hit_pa"]
    hits = batch["live_hit_hits"]
    minimum = batch.get("live_hit_pa_min", torch.ones_like(pa))
    minimum = torch.maximum(minimum, hits).clamp_max(logits.shape[1])
    support = torch.arange(1, logits.shape[1] + 1, device=logits.device)
    allowed_pa = torch.where(
        (pa >= 1)[:, None],
        support[None, :] == pa.clamp_max(logits.shape[1])[:, None],
        support[None, :] >= minimum[:, None],
    )
    hit_bucket = hits.clamp_max(logits.shape[2] - 1)
    selected = logits.gather(
        2, hit_bucket[:, None, None].expand(-1, logits.shape[1], 1)
    ).squeeze(-1)
    return torch.logsumexp(logits.flatten(1), dim=1) - torch.logsumexp(
        selected.masked_fill(~allowed_pa, -torch.inf), dim=1
    )


def _get(day: Any, name: str, default: Any = None) -> Any:
    if isinstance(day, Mapping):
        return day.get(name, default)
    return getattr(day, name, default)


def _float_array(value: Any, name: str, *, ndim: int) -> Any:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with {ndim} dimensions")
    return result


def _integer_vector(value: Any, name: str) -> Any:
    result = np.asarray(value)
    if result.ndim != 1 or (result.size and result.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return result.astype(np.int64, copy=False)


def _check_index(index: Any, count: int, name: str) -> None:
    if index.size and (index.min() < 0 or index.max() >= count):
        raise IndexError(f"{name} exceeds its day's node range")


def collate_kbo_day_graphs(
    days: Sequence[Any],
    *,
    device: Any = "cpu",
    max_pa_per_day: int | None = None,
    seed: int = 2026,
    max_edges_per_route_per_day: int | None = 20_000,
) -> dict[str, Any]:
    """Validate NumPy graphs once on CPU, then form a disjoint tensor union.

    PA sampling depends on (seed, day_id), not minibatch ordering. Edge caps
    retain the most recent events independently within each day/route. The
    returned CPU batch can be moved recursively by a trainer; tensor routes
    are dataclasses and carry no hidden CPU tensors.
    """

    torch, _ = require_torch()
    if not days:
        raise ValueError("cannot collate an empty day list")
    for name, limit in (
        ("max_pa_per_day", max_pa_per_day),
        ("max_edges_per_route_per_day", max_edges_per_route_per_day),
    ):
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError(f"{name} must be non-negative (0 is unlimited), or None")
    max_pa_per_day = max_pa_per_day or None
    max_edges_per_route_per_day = max_edges_per_route_per_day or None
    registry = kbo_route_registry()
    node_parts: dict[str, list[Any]] = {"player": [], "team": []}
    role_parts: dict[str, list[Any]] = {}
    box_feature_parts: dict[str, list[Any]] = {}
    box_feature_widths: dict[str, int] = {}
    node_graph_parts: dict[str, list[Any]] = {"player": [], "team": []}
    route_parts: dict[str, dict[str, list[Any]]] = {}
    vector_parts: dict[str, list[Any]] = {}
    matrix_parts: dict[str, list[Any]] = {
        "match_runs": [], "pa_context": [], "box_pa_counts": [], "box_pitch_targets": [],
    }
    box_pitch_masks: list[Any] = []
    query_ids: dict[str, list[str]] = {
        name: [] for name in ("match", "live_hit", "pa", "box_pa", "box_pitch")
    }
    day_ids: list[str] = []
    offsets = {"player": 0, "team": 0}
    node_widths: dict[str, int] = {}
    role_widths: dict[str, int] = {}
    route_widths: dict[str, int] = {}
    pa_context_width: int | None = None

    def append_vector(name: str, values: Any) -> None:
        vector_parts.setdefault(name, []).append(values)

    for graph_index, day in enumerate(days):
        day_id = str(_get(day, "day_id", graph_index))
        day_ids.append(day_id)
        nodes = _get(day, "node_features")
        if not isinstance(nodes, Mapping) or set(nodes) != {"player", "team"}:
            raise ValueError("each day needs player/team node_features")
        counts: dict[str, int] = {}
        for kind, raw in nodes.items():
            values = _float_array(raw, f"{kind} node_features", ndim=2)
            counts[kind] = values.shape[0]
            if (
                values.shape[1] < 1
                or node_widths.setdefault(kind, values.shape[1]) != values.shape[1]
            ):
                raise ValueError("node feature widths must be positive and consistent across days")
            node_parts[kind].append(values)
            node_graph_parts[kind].append(np.full(values.shape[0], graph_index, dtype=np.int64))
        roles = _get(day, "role_features", {})
        if not isinstance(roles, Mapping) or set(roles) - {"batting", "pitching"}:
            raise ValueError("role_features must contain only batting/pitching arrays")
        if graph_index and set(roles) != set(role_parts):
            raise ValueError("role feature keys must agree across days")
        for role, raw in roles.items():
            values = _float_array(raw, f"{role} role_features", ndim=2)
            if values.shape[0] != counts["player"]:
                raise ValueError("role features must have one row per player node")
            if role_widths.setdefault(role, values.shape[1]) != values.shape[1]:
                raise ValueError("role feature widths must agree across days")
            role_parts.setdefault(role, []).append(values)
        for kind in ("player", "team"):
            for role, default_width in (("batting", 19), ("pitching", 21)):
                name = f"{kind}_box_{role}_features"
                values = _float_array(
                    _get(day, name, np.zeros((counts[kind], default_width), dtype=np.float32)),
                    name, ndim=2,
                )
                if values.shape[0] != counts[kind] or values.shape[1] < 1:
                    raise ValueError(f"{name} must have one feature row per {kind} node")
                if box_feature_widths.setdefault(name, values.shape[1]) != values.shape[1]:
                    raise ValueError("box-score feature widths must agree across days")
                box_feature_parts.setdefault(name, []).append(values)
        routes = _get(day, "routes", {})
        if not isinstance(routes, Mapping):
            raise ValueError("routes must map reviewed names to numeric arrays")
        for name, raw_route in routes.items():
            route = registry.require(name)
            source = _integer_vector(raw_route["source_index"], f"{name} source_index")
            destination = _integer_vector(
                raw_route["destination_index"], f"{name} destination_index"
            )
            edge_count = source.size
            _check_index(source, counts[route.source_type], name)
            _check_index(destination, counts[route.destination_type], name)
            features = _float_array(raw_route["event_features"], f"{name} event_features", ndim=2)
            temporal = {
                key: _float_array(raw_route[key], f"{name} {key}", ndim=1)
                for key in ("event_age_seconds", "publication_delay_seconds", "weights")
            }
            if (
                destination.size != edge_count
                or features.shape[0] != edge_count
                or any(value.size != edge_count for value in temporal.values())
            ):
                raise ValueError("route columns must contain one value per edge")
            if route_widths.setdefault(name, features.shape[1]) != features.shape[1]:
                raise ValueError("route feature widths must agree across days")
            if np.any(temporal["weights"] < 0):
                raise ValueError("route weights cannot be negative")
            if np.any(temporal["event_age_seconds"] < temporal["publication_delay_seconds"] - 1e-6):
                raise ValueError("route contains information available after its cutoff")
            selected = np.arange(edge_count)
            if max_edges_per_route_per_day is not None and edge_count > max_edges_per_route_per_day:
                selected = np.argsort(temporal["event_age_seconds"], kind="stable")[
                    :max_edges_per_route_per_day
                ]
            columns = {
                "source_index": source[selected] + offsets[route.source_type],
                "destination_index": destination[selected] + offsets[route.destination_type],
                "event_features": features[selected],
                **{key: value[selected] for key, value in temporal.items()},
            }
            for key, values in columns.items():
                route_parts.setdefault(name, {}).setdefault(key, []).append(values)

        query_specs = {
            "match": (("match_home_team_index", "team"), ("match_away_team_index", "team")),
            "live_hit": (
                ("live_hit_player_index", "player"),
                ("live_hit_team_index", "team"),
                ("live_hit_opponent_index", "team"),
            ),
            "pa": (("pa_batter_index", "player"), ("pa_pitcher_index", "player")),
            "box_pa": (
                ("box_pa_player_index", "player"),
                ("box_pa_team_index", "team"),
                ("box_pa_opponent_index", "team"),
            ),
            "box_pitch": (
                ("box_pitch_player_index", "player"),
                ("box_pitch_team_index", "team"),
                ("box_pitch_opponent_index", "team"),
            ),
        }
        for task, specifications in query_specs.items():
            indices = {
                name: _integer_vector(_get(day, name, []), name) for name, _ in specifications
            }
            count = next(iter(indices.values())).size
            if any(index.size != count for index in indices.values()):
                raise ValueError(f"{task} query index columns must agree in length")
            for name, kind in specifications:
                _check_index(indices[name], counts[kind], name)
            selected = np.arange(count)
            if task == "pa" and max_pa_per_day is not None and count > max_pa_per_day:
                digest = hashlib.sha256(f"{seed}:{day_id}:pa".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                selected = np.sort(rng.choice(count, max_pa_per_day, replace=False))
            for name, kind in specifications:
                append_vector(name, indices[name][selected] + offsets[kind])
            append_vector(
                f"{task}_graph_index", np.full(selected.size, graph_index, dtype=np.int64)
            )
            ids = tuple(
                _get(
                    day, f"{task}_query_ids", (f"{day_id}:{task}:{index}" for index in range(count))
                )
            )
            if len(ids) != count:
                raise ValueError(f"{task}_query_ids must match query count")
            query_ids[task].extend(str(ids[index]) for index in selected)
            if task == "live_hit":
                pa_count = _integer_vector(_get(day, "live_hit_pa", []), "live_hit_pa")
                hits = _integer_vector(_get(day, "live_hit_hits", []), "live_hit_hits")
                minimum = _integer_vector(
                    _get(day, "live_hit_pa_min", np.maximum(pa_count, 1)), "live_hit_pa_min"
                )
                if (
                    pa_count.size != count
                    or hits.size != count
                    or minimum.size != count
                    or np.any((pa_count < 1) & (pa_count != -1))
                    or np.any(minimum < 1)
                    or np.any(hits < 0)
                    or np.any((pa_count >= 1) & ((hits > pa_count) | (minimum > pa_count)))
                ):
                    raise ValueError(
                        "Live Hit labels require PA >= 1 or unknown -1, positive PA minimum, "
                        "and 0 <= H <= known PA"
                    )
                append_vector("live_hit_pa", pa_count)
                append_vector("live_hit_pa_min", minimum)
                append_vector("live_hit_hits", hits)
            elif task in {"match", "pa"}:
                targets = _integer_vector(_get(day, f"{task}_targets", []), f"{task}_targets")
                classes = 3 if task == "match" else 10
                if targets.size != count or np.any(targets < 0) or np.any(targets >= classes):
                    raise ValueError(f"{task} labels must contain {classes}-class targets")
                append_vector(f"{task}_targets", targets[selected])
                if task == "match":
                    runs = _float_array(
                        _get(day, "match_runs", np.empty((0, 2))), "match_runs", ndim=2
                    )
                    if runs.shape != (count, 2) or np.any(runs < 0):
                        raise ValueError(
                            "match_runs must have non-negative home/away scores per game"
                        )
                    matrix_parts["match_runs"].append(runs)
                else:
                    context = _float_array(
                        _get(day, "pa_context", np.empty((count, 0))), "pa_context", ndim=2
                    )
                    if context.shape[0] != count:
                        raise ValueError("pa_context must have one row per PA query")
                    if pa_context_width is not None and context.shape[1] != pa_context_width:
                        raise ValueError("pa_context width must agree across days")
                    pa_context_width = context.shape[1]
                    matrix_parts["pa_context"].append(context[selected])
            elif task == "box_pa":
                counts_array = _float_array(
                    _get(day, "box_pa_counts", np.empty((0, 10))), "box_pa_counts", ndim=2
                )
                if (
                    counts_array.shape != (count, 10)
                    or np.any(counts_array < 0)
                    or np.any(counts_array != np.floor(counts_array))
                    or np.any(counts_array.sum(axis=1) < 1)
                ):
                    raise ValueError("box_pa_counts must contain observed integer outcome counts")
                matrix_parts["box_pa_counts"].append(counts_array)
            else:
                pitch_targets = _float_array(
                    _get(day, "box_pitch_targets", np.empty((0, 10))), "box_pitch_targets", ndim=2
                )
                raw_mask = np.asarray(_get(day, "box_pitch_mask", np.empty((0, 10), dtype=bool)))
                if (
                    pitch_targets.shape != (count, 10)
                    or raw_mask.shape != (count, 10)
                    or np.any((raw_mask != 0) & (raw_mask != 1))
                    or np.any(pitch_targets[raw_mask.astype(bool)] < 0)
                    or np.any(
                        pitch_targets[raw_mask.astype(bool)]
                        != np.floor(pitch_targets[raw_mask.astype(bool)])
                    )
                    or np.any(raw_mask.sum(axis=1) < 1)
                ):
                    raise ValueError("box_pitch requires observed integer counts and a valid mask")
                mask = raw_mask.astype(bool)
                # Placeholder zeros are explicitly masked, never target observations.
                matrix_parts["box_pitch_targets"].append(np.where(mask, pitch_targets, 0))
                box_pitch_masks.append(mask)
        for kind in offsets:
            offsets[kind] += counts[kind]

    def tensor(parts: Sequence[Any], *, integer: bool = False) -> Any:
        values = np.concatenate(parts, axis=0)
        return torch.as_tensor(
            values, dtype=torch.long if integer else torch.float32, device=device
        )

    tensors: dict[str, Any] = {
        "node_features": {kind: tensor(parts) for kind, parts in node_parts.items()},
        "role_features": {role: tensor(parts) for role, parts in role_parts.items()},
        "node_graph_index": {
            kind: tensor(parts, integer=True) for kind, parts in node_graph_parts.items()
        },
        "day_ids": tuple(day_ids),
        "_validated_on_cpu": True,
    }
    tensors.update({name: tensor(parts, integer=True) for name, parts in vector_parts.items()})
    tensors.update({name: tensor(parts) for name, parts in matrix_parts.items()})
    tensors.update({name: tensor(parts) for name, parts in box_feature_parts.items()})
    tensors["box_pitch_mask"] = torch.as_tensor(
        np.concatenate(box_pitch_masks, axis=0), dtype=torch.bool, device=device
    )
    tensors.update({f"{task}_query_ids": tuple(ids) for task, ids in query_ids.items()})
    torch_routes = []
    for name, columns in route_parts.items():
        route = registry.require(name)
        torch_routes.append(
            TorchAtomicRouteBatch(
                route_name=name,
                source_type=route.source_type,
                destination_type=route.destination_type,
                source_index=tensor(columns["source_index"], integer=True),
                destination_index=tensor(columns["destination_index"], integer=True),
                event_features=tensor(columns["event_features"]),
                event_age_seconds=tensor(columns["event_age_seconds"]),
                publication_delay_seconds=tensor(columns["publication_delay_seconds"]),
                weights=tensor(columns["weights"]),
                bidirectional=route.bidirectional,
            )
        )
    tensors["routes"] = tuple(torch_routes)
    return tensors


__all__ = [
    "KBO_ROUTE_NAMES",
    "KBORelGNNConfig",
    "KBORelGNNModel",
    "collate_kbo_day_graphs",
    "encode_live_hit_targets",
    "kbo_multitask_loss",
    "kbo_route_registry",
    "live_hit_observed_nll",
]
