"""Shared-backbone model boundary with task-specific adapters and heads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cpv26.models._torch import ModuleBase, nn, require_torch

from .contracts import (
    LIVE_HIT_TASK,
    MATCH_TASK,
    PA_TASK,
    LiveHitTaskBatch,
    MatchTaskBatch,
    PATaskBatch,
    TaskBatch,
)


class TaskSeparatedModel(ModuleBase):
    """One registered backbone plus independent task adapters and heads.

    ``backbone_inputs`` are passed to ``backbone`` as keyword arguments.  Each
    adapter receives the resulting shared state as its first argument and the
    batch's ``adapter_inputs`` as keyword arguments.  This keeps graph state
    selection outside the generic trainer while ensuring every task update is
    connected to the same registered backbone parameters.

    Adapter outputs have small, explicit contracts:

    * PA: a tensor, positional tuple, or keyword mapping accepted by ``pa_head``
    * Live Hit: one player-game embedding tensor
    * Match: a mapping containing ``home_embedding`` and ``away_embedding``;
      an optional ``game_context`` is forwarded to both match heads
    """

    def __init__(
        self,
        *,
        backbone: Any,
        pa_adapter: Any,
        live_hit_adapter: Any,
        match_adapter: Any,
        pa_head: Any,
        live_hit_head: Any,
        wdl_head: Any,
        run_head: Any,
    ) -> None:
        require_torch()
        super().__init__()
        modules = {
            "backbone": backbone,
            "pa_adapter": pa_adapter,
            "live_hit_adapter": live_hit_adapter,
            "match_adapter": match_adapter,
            "pa_head": pa_head,
            "live_hit_head": live_hit_head,
            "wdl_head": wdl_head,
            "run_head": run_head,
        }
        for name, module in modules.items():
            if not isinstance(module, nn.Module):
                raise TypeError(f"{name} must be a torch.nn.Module")
        self.backbone: Any = backbone
        self.pa_adapter: Any = pa_adapter
        self.live_hit_adapter: Any = live_hit_adapter
        self.match_adapter: Any = match_adapter
        self.pa_head: Any = pa_head
        self.live_hit_head: Any = live_hit_head
        self.wdl_head: Any = wdl_head
        self.run_head: Any = run_head

    def encode(self, backbone_inputs: Mapping[str, Any]) -> Any:
        """Run one task batch through the registered shared backbone."""

        return self.backbone(**dict(backbone_inputs))

    @staticmethod
    def _adapt(
        adapter: Any,
        shared_state: Any,
        adapter_inputs: Mapping[str, Any],
    ) -> Any:
        return adapter(shared_state, **dict(adapter_inputs))

    @staticmethod
    def _call_head(head: Any, adapted: Any) -> Any:
        if isinstance(adapted, Mapping):
            return head(**dict(adapted))
        if isinstance(adapted, tuple):
            return head(*adapted)
        return head(adapted)

    def pa_features(self, batch: PATaskBatch) -> Any:
        shared = self.encode(batch.backbone_inputs)
        return self._adapt(self.pa_adapter, shared, batch.adapter_inputs)

    def forward_pa(self, batch: PATaskBatch) -> Any:
        return self._call_head(self.pa_head, self.pa_features(batch))

    def live_hit_embedding(self, batch: LiveHitTaskBatch) -> Any:
        shared = self.encode(batch.backbone_inputs)
        adapted = self._adapt(self.live_hit_adapter, shared, batch.adapter_inputs)
        if isinstance(adapted, (Mapping, tuple)):
            raise TypeError("live_hit_adapter must return one player-game embedding tensor")
        return adapted

    def forward_live_hit(self, batch: LiveHitTaskBatch) -> dict[str, Any]:
        output = self.live_hit_head(self.live_hit_embedding(batch))
        if not isinstance(output, dict):
            raise TypeError("live_hit_head must return its declared output mapping")
        return output

    def match_features(self, batch: MatchTaskBatch) -> Mapping[str, Any]:
        shared = self.encode(batch.backbone_inputs)
        adapted = self._adapt(self.match_adapter, shared, batch.adapter_inputs)
        if not isinstance(adapted, Mapping):
            raise TypeError("match_adapter must return a mapping")
        required = {"home_embedding", "away_embedding"}
        missing = sorted(required.difference(adapted))
        unknown = sorted(set(adapted).difference((*required, "game_context")))
        if missing:
            raise KeyError(f"match_adapter output is missing: {', '.join(missing)}")
        if unknown:
            raise KeyError(f"unknown match_adapter outputs: {', '.join(unknown)}")
        return adapted

    def forward_match(self, batch: MatchTaskBatch) -> dict[str, Any]:
        features = dict(self.match_features(batch))
        home_embedding = features.pop("home_embedding")
        away_embedding = features.pop("away_embedding")
        return {
            "wdl_logits": self.wdl_head(
                home_embedding,
                away_embedding,
                **features,
            ),
            "run_parameters": self.run_head(
                home_embedding,
                away_embedding,
                **features,
            ),
        }

    def forward(self, task: str, batch: TaskBatch) -> Any:
        """Dispatch inference without conflating the three target contracts."""

        if task == PA_TASK:
            if not isinstance(batch, PATaskBatch):
                raise TypeError("pa task requires PATaskBatch")
            return self.forward_pa(batch)
        if task == LIVE_HIT_TASK:
            if not isinstance(batch, LiveHitTaskBatch):
                raise TypeError("live_hit task requires LiveHitTaskBatch")
            return self.forward_live_hit(batch)
        if task == MATCH_TASK:
            if not isinstance(batch, MatchTaskBatch):
                raise TypeError("match task requires MatchTaskBatch")
            return self.forward_match(batch)
        raise KeyError(f"unknown training task: {task}")


__all__ = ["TaskSeparatedModel"]
