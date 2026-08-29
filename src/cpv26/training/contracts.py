"""In-memory tensor contracts for the three supervised prediction tasks.

These contracts deliberately start *after* snapshot/materialization.  They do
not imply a Parquet layout, a provider, or a particular ``DataLoader``.  A
loader only needs to yield the corresponding task batch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from cpv26.models._torch import require_torch

PA_TASK = "pa"
LIVE_HIT_TASK = "live_hit"
MATCH_TASK = "match"
TASK_NAMES: tuple[str, str, str] = (PA_TASK, LIVE_HIT_TASK, MATCH_TASK)


def _integer_tensor(value: Any, *, name: str) -> Any:
    torch, _ = require_torch()
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch tensor")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must be an integer tensor")
    return value


def _binary_tensor(value: Any, *, name: str) -> Any:
    torch, _ = require_torch()
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch tensor")
    if value.dtype != torch.bool:
        value = _integer_tensor(value, name=name)
        if bool(((value != 0) & (value != 1)).any().item()):
            raise ValueError(f"{name} must contain only 0 or 1")
    return value


def _non_negative(value: Any, *, name: str) -> None:
    if bool((value < 0).any().item()):
        raise ValueError(f"{name} must be non-negative")


def _batch_vector(value: Any, *, name: str) -> None:
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional batch tensor")


def _same_shape(*values: Any) -> tuple[int, ...]:
    shapes = {tuple(value.shape) for value in values}
    if len(shapes) != 1:
        raise ValueError("all target tensors in a task batch must have identical shapes")
    return tuple(values[0].shape)


def _frozen_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType(dict(value))


def _move_value(value: Any, device: Any) -> Any:
    torch, _ = require_torch()
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _move_value(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_value(item, device) for item in value)
    if isinstance(value, list):
        return [_move_value(item, device) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class PATargets:
    """One ten-way neural class index per historical plate appearance.

    Build these indices with ``cpv26.simulation.neural_training_target_index``;
    catcher-interference rows train the adapter rate and are not PA-head rows.
    """

    outcome_index: Any

    def __post_init__(self) -> None:
        outcome_index = _integer_tensor(self.outcome_index, name="outcome_index")
        _batch_vector(outcome_index, name="outcome_index")
        _non_negative(outcome_index, name="outcome_index")

    @property
    def sample_count(self) -> int:
        return int(self.outcome_index.numel())

    def to(self, device: Any) -> PATargets:
        return PATargets(outcome_index=_move_value(self.outcome_index, device))


@dataclass(frozen=True, slots=True)
class LiveHitTargets:
    """Resolved player-game labels for the conditional Live Hit model.

    ``appeared`` means that the player recorded at least one plate appearance.
    ``game_played`` separates a cancelled/no-result game from a played game in
    which the player stayed on the bench.  ``started`` is the realised starting
    lineup label, not a projected-lineup feature.

    ``label_observed`` is deliberately required.  False rows use zero/false
    placeholders and never contribute to the loss.  Cancelled/no-result rows
    are observed rows with ``game_played=False`` and also never contribute to
    the conditional ``P(PA, H | game played)`` loss.  A separate game-status
    model must provide ``P(game played)`` at inference time.
    """

    appeared: Any
    plate_appearances: Any
    hits: Any
    game_played: Any
    started: Any
    label_observed: Any

    def __post_init__(self) -> None:
        torch, _ = require_torch()
        appeared = _binary_tensor(self.appeared, name="appeared")
        game_played = _binary_tensor(self.game_played, name="game_played")
        started = _binary_tensor(self.started, name="started")
        label_observed = _binary_tensor(self.label_observed, name="label_observed")
        plate_appearances = _integer_tensor(
            self.plate_appearances,
            name="plate_appearances",
        )
        hits = _integer_tensor(self.hits, name="hits")
        _same_shape(
            appeared,
            plate_appearances,
            hits,
            game_played,
            started,
            label_observed,
        )
        _batch_vector(appeared, name="appeared")
        _non_negative(plate_appearances, name="plate_appearances")
        _non_negative(hits, name="hits")
        if bool((hits > plate_appearances).any().item()):
            raise ValueError("hits cannot exceed plate appearances")
        has_plate_appearance = plate_appearances > 0
        appearance_mismatch = (
            appeared.to(dtype=has_plate_appearance.dtype) != has_plate_appearance
        )
        if bool(appearance_mismatch.any().item()):
            raise ValueError("appeared must be true exactly when plate_appearances is positive")
        appeared_mask = appeared.to(dtype=torch.bool)
        game_played_mask = game_played.to(dtype=torch.bool)
        started_mask = started.to(dtype=torch.bool)
        if bool((started_mask & ~game_played_mask).any().item()):
            raise ValueError("started requires game_played=true")
        if bool((appeared_mask & ~game_played_mask).any().item()):
            raise ValueError("appeared requires game_played=true")
        unobserved = ~label_observed.to(dtype=torch.bool)
        has_unobserved_placeholder_violation = (
            game_played_mask
            | started_mask
            | appeared_mask
            | (plate_appearances != 0)
            | (hits != 0)
        ) & unobserved
        if bool(has_unobserved_placeholder_violation.any().item()):
            raise ValueError(
                "unobserved Live Hit labels must use false/zero placeholders"
            )

    @property
    def joint_loss_mask(self) -> Any:
        """Rows valid for ``P(PA, H | game played)`` supervision."""

        torch, _ = require_torch()
        return self.label_observed.to(dtype=torch.bool) & self.game_played.to(
            dtype=torch.bool
        )

    @property
    def supervised_sample_count(self) -> int:
        return int(self.joint_loss_mask.sum().item())

    @property
    def sample_count(self) -> int:
        return int(self.plate_appearances.numel())

    def to(self, device: Any) -> LiveHitTargets:
        return LiveHitTargets(
            appeared=_move_value(self.appeared, device),
            plate_appearances=_move_value(self.plate_appearances, device),
            hits=_move_value(self.hits, device),
            game_played=_move_value(self.game_played, device),
            started=_move_value(self.started, device),
            label_observed=_move_value(self.label_observed, device),
        )


@dataclass(frozen=True, slots=True)
class MatchTargets:
    """Terminal game labels with an explicit completed-result mask.

    ``result_observed`` means that the terminal status is known.  A cancelled,
    postponed, or official no-result row therefore has
    ``result_observed=True`` and ``completed=False``.  An unresolved future row
    has both flags false.  Every non-completed row must use ``-1`` sentinels for
    WDL and scores so that it cannot silently masquerade as a 0:0 draw.
    """

    wdl_class: Any
    home_runs: Any
    away_runs: Any
    completed: Any
    result_observed: Any

    def __post_init__(self) -> None:
        torch, _ = require_torch()
        wdl_class = _integer_tensor(self.wdl_class, name="wdl_class")
        home_runs = _integer_tensor(self.home_runs, name="home_runs")
        away_runs = _integer_tensor(self.away_runs, name="away_runs")
        completed = _binary_tensor(self.completed, name="completed")
        result_observed = _binary_tensor(
            self.result_observed,
            name="result_observed",
        )
        _same_shape(
            wdl_class,
            home_runs,
            away_runs,
            completed,
            result_observed,
        )
        _batch_vector(wdl_class, name="wdl_class")
        completed_mask = completed.to(dtype=torch.bool)
        observed_mask = result_observed.to(dtype=torch.bool)
        if bool((completed_mask & ~observed_mask).any().item()):
            raise ValueError("completed games require result_observed=true")
        valid_wdl = wdl_class[completed_mask]
        valid_home_runs = home_runs[completed_mask]
        valid_away_runs = away_runs[completed_mask]
        if bool((valid_home_runs < 0).any().item()):
            raise ValueError("completed home_runs must be non-negative")
        if bool((valid_away_runs < 0).any().item()):
            raise ValueError("completed away_runs must be non-negative")
        if bool(((valid_wdl < 0) | (valid_wdl > 2)).any().item()):
            raise ValueError("wdl_class must use 0=away_win, 1=draw, 2=home_win")
        derived = torch.where(
            valid_home_runs > valid_away_runs,
            torch.full_like(valid_wdl, 2),
            torch.where(
                valid_home_runs == valid_away_runs,
                torch.ones_like(valid_wdl),
                torch.zeros_like(valid_wdl),
            ),
        )
        if bool((valid_wdl != derived).any().item()):
            raise ValueError("wdl_class is inconsistent with the home/away run targets")
        invalid_mask = ~completed_mask
        invalid_labels_are_sentinels = (
            (wdl_class == -1) & (home_runs == -1) & (away_runs == -1)
        )
        if bool((invalid_mask & ~invalid_labels_are_sentinels).any().item()):
            raise ValueError("non-completed games must use -1 result sentinels")

    @property
    def result_loss_mask(self) -> Any:
        """Rows carrying an official result eligible for match supervision."""

        torch, _ = require_torch()
        return self.completed.to(dtype=torch.bool) & self.result_observed.to(
            dtype=torch.bool
        )

    @property
    def supervised_sample_count(self) -> int:
        return int(self.result_loss_mask.sum().item())

    @property
    def sample_count(self) -> int:
        return int(self.wdl_class.numel())

    def to(self, device: Any) -> MatchTargets:
        return MatchTargets(
            wdl_class=_move_value(self.wdl_class, device),
            home_runs=_move_value(self.home_runs, device),
            away_runs=_move_value(self.away_runs, device),
            completed=_move_value(self.completed, device),
            result_observed=_move_value(self.result_observed, device),
        )


@dataclass(frozen=True, slots=True)
class PATaskBatch:
    """Materialized PA inputs and targets yielded by a PA-specific loader."""

    backbone_inputs: Mapping[str, Any]
    targets: PATargets
    adapter_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backbone_inputs",
            _frozen_mapping(self.backbone_inputs, name="backbone_inputs"),
        )
        object.__setattr__(
            self,
            "adapter_inputs",
            _frozen_mapping(self.adapter_inputs, name="adapter_inputs"),
        )

    def to(self, device: Any) -> PATaskBatch:
        return PATaskBatch(
            backbone_inputs=_move_value(self.backbone_inputs, device),
            adapter_inputs=_move_value(self.adapter_inputs, device),
            targets=self.targets.to(device),
        )


@dataclass(frozen=True, slots=True)
class LiveHitTaskBatch:
    """Materialized player-game inputs and Live Hit targets."""

    backbone_inputs: Mapping[str, Any]
    targets: LiveHitTargets
    adapter_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backbone_inputs",
            _frozen_mapping(self.backbone_inputs, name="backbone_inputs"),
        )
        object.__setattr__(
            self,
            "adapter_inputs",
            _frozen_mapping(self.adapter_inputs, name="adapter_inputs"),
        )

    def to(self, device: Any) -> LiveHitTaskBatch:
        return LiveHitTaskBatch(
            backbone_inputs=_move_value(self.backbone_inputs, device),
            adapter_inputs=_move_value(self.adapter_inputs, device),
            targets=self.targets.to(device),
        )


@dataclass(frozen=True, slots=True)
class MatchTaskBatch:
    """Materialized game inputs and match prediction targets."""

    backbone_inputs: Mapping[str, Any]
    targets: MatchTargets
    adapter_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backbone_inputs",
            _frozen_mapping(self.backbone_inputs, name="backbone_inputs"),
        )
        object.__setattr__(
            self,
            "adapter_inputs",
            _frozen_mapping(self.adapter_inputs, name="adapter_inputs"),
        )

    def to(self, device: Any) -> MatchTaskBatch:
        return MatchTaskBatch(
            backbone_inputs=_move_value(self.backbone_inputs, device),
            adapter_inputs=_move_value(self.adapter_inputs, device),
            targets=self.targets.to(device),
        )


TaskBatch = PATaskBatch | LiveHitTaskBatch | MatchTaskBatch


__all__ = [
    "LIVE_HIT_TASK",
    "MATCH_TASK",
    "PA_TASK",
    "TASK_NAMES",
    "LiveHitTargets",
    "LiveHitTaskBatch",
    "MatchTargets",
    "MatchTaskBatch",
    "PATargets",
    "PATaskBatch",
    "TaskBatch",
]
