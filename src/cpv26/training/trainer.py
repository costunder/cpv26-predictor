"""Deterministic alternating optimization over three task-specific loaders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

from cpv26.models._torch import require_torch

from .contracts import TASK_NAMES, TaskBatch
from .losses import MultiTaskLossComposer, TaskLossConfig
from .model import TaskSeparatedModel


@dataclass(frozen=True, slots=True)
class CheckpointLineage:
    """Immutable compatibility identity for materialized features and labels."""

    feature_version: str
    route_version: str
    label_schema_version: str
    model_version: str

    def __post_init__(self) -> None:
        values = {
            "feature_version": self.feature_version,
            "route_version": self.route_version,
            "label_schema_version": self.label_schema_version,
            "model_version": self.model_version,
        }
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty version string")
            if value != value.strip():
                raise ValueError(f"{name} cannot have surrounding whitespace")


@dataclass(frozen=True, slots=True)
class TaskStepRecord:
    """Scalar diagnostics for one optimizer update."""

    task: str
    global_step: int
    loss: float
    components: Mapping[str, float]
    sample_count: int


class AlternatingMultiTaskTrainer:
    """Alternate finite task loaders without combining incompatible rows.

    The fixed ``task_order`` is repeated and exhausted tasks are skipped.  A
    repeated name can intentionally give a task more updates per round (for
    example ``("pa", "pa", "live_hit", "match")``).  Loader shuffling remains
    the caller's responsibility; the trainer seeds torch before iterator
    creation so ordinary seeded DataLoaders remain reproducible by epoch.
    """

    checkpoint_version = 1

    def __init__(
        self,
        *,
        model: TaskSeparatedModel,
        optimizer: Any,
        loss_composer: MultiTaskLossComposer | None = None,
        task_order: tuple[str, ...] = TASK_NAMES,
        device: Any = "cpu",
        seed: int = 0,
        gradient_clip_norm: float | None = None,
        checkpoint_lineage: CheckpointLineage | None = None,
    ) -> None:
        torch, _ = require_torch()
        if not isinstance(model, TaskSeparatedModel):
            raise TypeError("model must be TaskSeparatedModel")
        if not task_order:
            raise ValueError("task_order cannot be empty")
        unknown = sorted(set(task_order).difference(TASK_NAMES))
        missing = sorted(set(TASK_NAMES).difference(task_order))
        if unknown:
            raise ValueError(f"task_order contains unknown tasks: {', '.join(unknown)}")
        if missing:
            raise ValueError(f"task_order must include: {', '.join(missing)}")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if not hasattr(optimizer, "step") or not hasattr(optimizer, "zero_grad"):
            raise TypeError("optimizer must implement the torch optimizer contract")

        self.model: Any = model
        self.optimizer = optimizer
        self.loss_composer = loss_composer or MultiTaskLossComposer()
        self.task_order = tuple(task_order)
        self.device = torch.device(device)
        self.seed = seed
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_lineage = checkpoint_lineage
        self.epoch = 0
        self.global_step = 0
        self.task_steps = {task: 0 for task in TASK_NAMES}
        cast(Any, self.model).to(self.device)

    @staticmethod
    def _validate_loaders(loaders: Mapping[str, Iterable[TaskBatch]]) -> None:
        missing = sorted(set(TASK_NAMES).difference(loaders))
        unknown = sorted(set(loaders).difference(TASK_NAMES))
        if missing:
            raise KeyError(f"missing task loaders: {', '.join(missing)}")
        if unknown:
            raise KeyError(f"unknown task loaders: {', '.join(unknown)}")

    def _step(self, task: str, raw_batch: TaskBatch) -> TaskStepRecord:
        batch = raw_batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_composer(self.model, task, batch)
        if loss.total.ndim != 0:
            raise ValueError("task loss total must be a scalar tensor")
        if not bool(loss.total.isfinite().item()):
            raise FloatingPointError(f"non-finite {task} loss")
        loss.total.backward()
        if self.gradient_clip_norm is not None:
            torch, _ = require_torch()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
            )
        self.optimizer.step()
        self.global_step += 1
        self.task_steps[task] += 1
        return TaskStepRecord(
            task=task,
            global_step=self.global_step,
            loss=float(loss.total.detach().cpu().item()),
            components={
                name: float(value.detach().cpu().item())
                for name, value in loss.components.items()
            },
            sample_count=loss.sample_count,
        )

    def train_epoch(
        self,
        loaders: Mapping[str, Iterable[TaskBatch]],
    ) -> tuple[TaskStepRecord, ...]:
        """Consume each finite loader once using the fixed alternating order."""

        torch, _ = require_torch()
        self._validate_loaders(loaders)
        torch.manual_seed(self.seed + self.epoch)
        iterators = {task: iter(loaders[task]) for task in TASK_NAMES}
        active = set(TASK_NAMES)
        records: list[TaskStepRecord] = []
        self.model.train()

        while active:
            progressed = False
            for task in self.task_order:
                if task not in active:
                    continue
                try:
                    raw_batch = next(iterators[task])
                except StopIteration:
                    active.remove(task)
                    continue
                if not hasattr(raw_batch, "to"):
                    raise TypeError(f"{task} loader must yield its task batch contract")
                records.append(self._step(task, raw_batch))
                progressed = True
            if not progressed and active:
                raise RuntimeError("task scheduling made no progress")

        self.epoch += 1
        return tuple(records)

    def checkpoint_state(self) -> dict[str, Any]:
        """Return model, optimizer, trainer, and RNG state for epoch-boundary resume."""

        torch, _ = require_torch()
        state: dict[str, Any] = {
            "checkpoint_version": self.checkpoint_version,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "trainer": {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "task_steps": dict(self.task_steps),
                "task_order": self.task_order,
                "seed": self.seed,
                "gradient_clip_norm": self.gradient_clip_norm,
            },
            "loss_config": asdict(self.loss_composer.config),
            "checkpoint_lineage": (
                asdict(self.checkpoint_lineage)
                if self.checkpoint_lineage is not None
                else None
            ),
            "torch_rng_state": torch.get_rng_state().clone(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = [
                value.clone() for value in torch.cuda.get_rng_state_all()
            ]
        return state

    def load_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        """Restore a state returned by :meth:`checkpoint_state`."""

        torch, _ = require_torch()
        if state.get("checkpoint_version") != self.checkpoint_version:
            raise ValueError("unsupported multi-task checkpoint version")
        raw_lineage = state.get("checkpoint_lineage")
        restored_lineage = (
            CheckpointLineage(**raw_lineage) if raw_lineage is not None else None
        )
        if restored_lineage != self.checkpoint_lineage:
            raise ValueError(
                "checkpoint lineage does not match the configured feature, route, "
                "label, and model versions"
            )
        self.model.load_state_dict(state["model"], strict=strict)
        self.optimizer.load_state_dict(state["optimizer"])
        trainer_state = state["trainer"]
        restored_order = tuple(trainer_state["task_order"])
        unknown = set(restored_order).difference(TASK_NAMES)
        missing = set(TASK_NAMES).difference(restored_order)
        if unknown or missing:
            raise ValueError("checkpoint contains an invalid task order")
        self.epoch = int(trainer_state["epoch"])
        self.global_step = int(trainer_state["global_step"])
        restored_steps = dict(trainer_state["task_steps"])
        if set(restored_steps) != set(TASK_NAMES):
            raise ValueError("checkpoint contains invalid per-task step counters")
        self.task_steps = {task: int(restored_steps[task]) for task in TASK_NAMES}
        self.task_order = restored_order
        self.seed = int(trainer_state["seed"])
        self.gradient_clip_norm = trainer_state["gradient_clip_norm"]
        self.loss_composer = MultiTaskLossComposer(TaskLossConfig(**state["loss_config"]))
        torch.set_rng_state(state["torch_rng_state"])
        if "cuda_rng_state_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


__all__ = ["AlternatingMultiTaskTrainer", "CheckpointLineage", "TaskStepRecord"]
