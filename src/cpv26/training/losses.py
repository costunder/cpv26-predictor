"""Task-specific supervised losses over one shared-backbone model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cpv26.models._torch import require_torch

from .contracts import (
    LIVE_HIT_TASK,
    MATCH_TASK,
    PA_TASK,
    LiveHitTaskBatch,
    MatchTaskBatch,
    PATaskBatch,
    TaskBatch,
)
from .model import TaskSeparatedModel


@dataclass(frozen=True, slots=True)
class TaskLossConfig:
    """Weights for losses that are meaningful within each task granularity."""

    pa_cross_entropy: float = 1.0
    live_hit_joint_nll: float = 1.0
    match_wdl_cross_entropy: float = 1.0
    match_run_nll: float = 1.0

    def __post_init__(self) -> None:
        values = {
            "pa_cross_entropy": self.pa_cross_entropy,
            "live_hit_joint_nll": self.live_hit_joint_nll,
            "match_wdl_cross_entropy": self.match_wdl_cross_entropy,
            "match_run_nll": self.match_run_nll,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.pa_cross_entropy == 0.0:
            raise ValueError("pa_cross_entropy must be positive")
        if self.live_hit_joint_nll == 0.0:
            raise ValueError("live_hit_joint_nll must be positive")
        if self.match_wdl_cross_entropy + self.match_run_nll == 0.0:
            raise ValueError("at least one match loss weight must be positive")


@dataclass(frozen=True, slots=True)
class TaskLoss:
    """Differentiable total plus named, unweighted diagnostic components."""

    task: str
    total: Any
    components: Mapping[str, Any]
    sample_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))


class MultiTaskLossComposer:
    """Compose the correct loss for a batch without mixing row granularities."""

    def __init__(self, config: TaskLossConfig | None = None) -> None:
        self.config = config or TaskLossConfig()

    @staticmethod
    def _check_logit_target_shape(logits: Any, target: Any, *, name: str) -> None:
        expected_shape = (*target.shape, logits.shape[-1])
        if logits.shape != expected_shape:
            raise ValueError(
                f"{name} logits must have shape {expected_shape}; "
                f"received {tuple(logits.shape)}"
            )
        if bool((target >= logits.shape[-1]).any().item()):
            raise ValueError(f"{name} target index exceeds the declared class count")

    def pa_loss(self, model: TaskSeparatedModel, batch: PATaskBatch) -> TaskLoss:
        torch, _ = require_torch()
        logits = model.forward_pa(batch)
        target = batch.targets.outcome_index.to(device=logits.device, dtype=torch.long)
        self._check_logit_target_shape(logits, target, name="PA")
        cross_entropy = torch.nn.functional.cross_entropy(logits, target)
        return TaskLoss(
            task=PA_TASK,
            total=self.config.pa_cross_entropy * cross_entropy,
            components={"cross_entropy": cross_entropy},
            sample_count=batch.targets.sample_count,
        )

    def live_hit_loss(
        self,
        model: TaskSeparatedModel,
        batch: LiveHitTaskBatch,
    ) -> TaskLoss:
        require_torch()
        embedding = model.live_hit_embedding(batch)
        per_sample_nll = model.live_hit_head.negative_log_likelihood(
            embedding,
            batch.targets.plate_appearances,
            batch.targets.hits,
            reduction="none",
        )
        loss_mask = batch.targets.joint_loss_mask.to(device=per_sample_nll.device)
        if per_sample_nll.shape != loss_mask.shape:
            raise ValueError(
                "Live Hit loss mask must match the per-player-game loss shape"
            )
        supervised_sample_count = int(loss_mask.sum().item())
        if supervised_sample_count == 0:
            raise ValueError(
                "Live Hit batch has no observed played-game labels for supervision"
            )
        joint_nll = per_sample_nll[loss_mask].mean()
        return TaskLoss(
            task=LIVE_HIT_TASK,
            total=self.config.live_hit_joint_nll * joint_nll,
            components={"joint_nll": joint_nll},
            sample_count=supervised_sample_count,
        )

    def match_loss(self, model: TaskSeparatedModel, batch: MatchTaskBatch) -> TaskLoss:
        torch, _ = require_torch()
        output = model.forward_match(batch)
        wdl_logits = output["wdl_logits"]
        run_parameters = output["run_parameters"]
        result_mask = batch.targets.result_loss_mask.to(device=wdl_logits.device)
        expected_result_shape = tuple(batch.targets.wdl_class.shape)
        if tuple(result_mask.shape) != expected_result_shape:
            raise ValueError("match result mask must match the target shape")
        supervised_sample_count = int(result_mask.sum().item())
        if supervised_sample_count == 0:
            raise ValueError("match batch has no completed observed results for supervision")
        wdl_target = batch.targets.wdl_class.to(
            device=wdl_logits.device,
            dtype=torch.long,
        )[result_mask]
        supervised_wdl_logits = wdl_logits[result_mask]
        self._check_logit_target_shape(
            supervised_wdl_logits,
            wdl_target,
            name="WDL",
        )
        wdl_cross_entropy = torch.nn.functional.cross_entropy(
            supervised_wdl_logits,
            wdl_target,
        )

        required_run_parameters = {
            "home_mean",
            "away_mean",
            "home_dispersion",
            "away_dispersion",
        }
        if not isinstance(run_parameters, Mapping):
            raise TypeError("run head must return a mapping")
        missing = sorted(required_run_parameters.difference(run_parameters))
        if missing:
            raise KeyError(f"run head output is missing: {', '.join(missing)}")
        reference = run_parameters["home_mean"]
        expected_run_shape = expected_result_shape
        for name in required_run_parameters:
            if run_parameters[name].shape != expected_run_shape:
                raise ValueError(
                    f"run parameter {name} must have shape {expected_run_shape}"
                )
        run_mask = result_mask.to(device=reference.device)
        home_runs = batch.targets.home_runs.to(device=reference.device)[run_mask]
        away_runs = batch.targets.away_runs.to(device=reference.device)[run_mask]
        home_log_prob = model.run_head.negative_binomial_log_prob(
            home_runs,
            run_parameters["home_mean"][run_mask],
            run_parameters["home_dispersion"][run_mask],
        )
        away_log_prob = model.run_head.negative_binomial_log_prob(
            away_runs,
            run_parameters["away_mean"][run_mask],
            run_parameters["away_dispersion"][run_mask],
        )
        run_nll = -(home_log_prob + away_log_prob).mean()
        total = (
            self.config.match_wdl_cross_entropy * wdl_cross_entropy
            + self.config.match_run_nll * run_nll
        )
        return TaskLoss(
            task=MATCH_TASK,
            total=total,
            components={
                "wdl_cross_entropy": wdl_cross_entropy,
                "run_nll": run_nll,
            },
            sample_count=supervised_sample_count,
        )

    def __call__(
        self,
        model: TaskSeparatedModel,
        task: str,
        batch: TaskBatch,
    ) -> TaskLoss:
        if task == PA_TASK:
            if not isinstance(batch, PATaskBatch):
                raise TypeError("pa task requires PATaskBatch")
            return self.pa_loss(model, batch)
        if task == LIVE_HIT_TASK:
            if not isinstance(batch, LiveHitTaskBatch):
                raise TypeError("live_hit task requires LiveHitTaskBatch")
            return self.live_hit_loss(model, batch)
        if task == MATCH_TASK:
            if not isinstance(batch, MatchTaskBatch):
                raise TypeError("match task requires MatchTaskBatch")
            return self.match_loss(model, batch)
        raise KeyError(f"unknown training task: {task}")


__all__ = ["MultiTaskLossComposer", "TaskLoss", "TaskLossConfig"]
