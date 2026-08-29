"""Task-separated tensor contracts, losses, model boundary, and trainer.

The package is optional-PyTorch safe at import time.  It intentionally does
not provide provider-specific extraction or file-loading implementations.
"""

from .contracts import (
    LIVE_HIT_TASK,
    MATCH_TASK,
    PA_TASK,
    TASK_NAMES,
    LiveHitTargets,
    LiveHitTaskBatch,
    MatchTargets,
    MatchTaskBatch,
    PATargets,
    PATaskBatch,
    TaskBatch,
)
from .losses import MultiTaskLossComposer, TaskLoss, TaskLossConfig
from .model import TaskSeparatedModel
from .trainer import AlternatingMultiTaskTrainer, CheckpointLineage, TaskStepRecord

__all__ = [
    "LIVE_HIT_TASK",
    "MATCH_TASK",
    "PA_TASK",
    "TASK_NAMES",
    "AlternatingMultiTaskTrainer",
    "CheckpointLineage",
    "LiveHitTargets",
    "LiveHitTaskBatch",
    "MatchTargets",
    "MatchTaskBatch",
    "MultiTaskLossComposer",
    "PATargets",
    "PATaskBatch",
    "TaskBatch",
    "TaskLoss",
    "TaskLossConfig",
    "TaskSeparatedModel",
    "TaskStepRecord",
]
