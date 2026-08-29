from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from cpv26.models import torch_available
from cpv26.training import (
    LIVE_HIT_TASK,
    MATCH_TASK,
    PA_TASK,
    AlternatingMultiTaskTrainer,
    CheckpointLineage,
    LiveHitTargets,
    LiveHitTaskBatch,
    MatchTargets,
    MatchTaskBatch,
    MultiTaskLossComposer,
    PATargets,
    PATaskBatch,
    TaskSeparatedModel,
)

requires_torch = pytest.mark.skipif(
    not torch_available(),
    reason="PyTorch optional dependency missing",
)


def test_training_package_import_is_safe_without_torch() -> None:
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

        import cpv26.training as training
        from cpv26.models import TorchUnavailableError

        assert training.TASK_NAMES == ("pa", "live_hit", "match")
        try:
            training.PATargets(outcome_index=[0])
        except TorchUnavailableError:
            pass
        else:
            raise AssertionError("tensor contract succeeded without PyTorch")
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


@requires_torch
def test_player_game_head_learns_pa_conditional_hit_logits() -> None:
    import torch

    from cpv26.models import DirectPlayerGameHead

    head = DirectPlayerGameHead(
        3,
        hidden_dim=6,
        max_plate_appearances=4,
        max_hits=3,
        dropout=0.0,
    )
    with torch.no_grad():
        head.hits.weight.zero_()
        head.hits.bias.zero_()
        hit_bucket_count = head.max_hits + 2
        head.hits.bias[1 * hit_bucket_count + 1] = 8.0
        head.hits.bias[2 * hit_bucket_count + 2] = 8.0

    output = head(torch.zeros((2, 3)))
    probabilities = head.probabilities(torch.zeros((2, 3)))
    conditional = probabilities["conditional_hit_probabilities"]

    assert output["hit_count_logits"].shape == (2, 6, 5)
    assert conditional[0, 1, 1] > 0.99
    assert conditional[0, 2, 2] > 0.99
    assert torch.count_nonzero(conditional[:, 0, 1:]) == 0
    for pa_count in range(1, 5):
        for hit_count in range(4):
            if hit_count > pa_count:
                assert torch.count_nonzero(conditional[:, pa_count, hit_count]) == 0


@requires_torch
def test_task_target_contracts_reject_cross_task_inconsistency() -> None:
    import torch

    valid = LiveHitTargets(
        appeared=torch.tensor([False, True]),
        plate_appearances=torch.tensor([0, 4]),
        hits=torch.tensor([0, 2]),
        game_played=torch.tensor([True, True]),
        started=torch.tensor([False, True]),
        label_observed=torch.tensor([True, True]),
    )
    assert valid.sample_count == 2
    assert valid.supervised_sample_count == 2

    with pytest.raises(ValueError, match="appeared must be true"):
        LiveHitTargets(
            appeared=torch.tensor([False]),
            plate_appearances=torch.tensor([1]),
            hits=torch.tensor([0]),
            game_played=torch.tensor([True]),
            started=torch.tensor([False]),
            label_observed=torch.tensor([True]),
        )
    with pytest.raises(ValueError, match="hits cannot exceed"):
        LiveHitTargets(
            appeared=torch.tensor([True]),
            plate_appearances=torch.tensor([2]),
            hits=torch.tensor([3]),
            game_played=torch.tensor([True]),
            started=torch.tensor([True]),
            label_observed=torch.tensor([True]),
        )
    with pytest.raises(ValueError, match="inconsistent"):
        MatchTargets(
            wdl_class=torch.tensor([0]),
            home_runs=torch.tensor([5]),
            away_runs=torch.tensor([2]),
            completed=torch.tensor([True]),
            result_observed=torch.tensor([True]),
        )


@requires_torch
def test_live_hit_contract_separates_cancelled_unobserved_and_bench_rows() -> None:
    import torch

    targets = LiveHitTargets(
        appeared=torch.tensor([True, False, False, False]),
        plate_appearances=torch.tensor([4, 0, 0, 0]),
        hits=torch.tensor([2, 0, 0, 0]),
        game_played=torch.tensor([True, False, False, True]),
        started=torch.tensor([True, False, False, False]),
        label_observed=torch.tensor([True, True, False, True]),
    )

    assert torch.equal(
        targets.joint_loss_mask,
        torch.tensor([True, False, False, True]),
    )
    assert targets.sample_count == 4
    assert targets.supervised_sample_count == 2

    with pytest.raises(ValueError, match="unobserved Live Hit labels"):
        LiveHitTargets(
            appeared=torch.tensor([False]),
            plate_appearances=torch.tensor([0]),
            hits=torch.tensor([0]),
            game_played=torch.tensor([True]),
            started=torch.tensor([False]),
            label_observed=torch.tensor([False]),
        )
    with pytest.raises(ValueError, match="appeared requires game_played"):
        LiveHitTargets(
            appeared=torch.tensor([True]),
            plate_appearances=torch.tensor([1]),
            hits=torch.tensor([0]),
            game_played=torch.tensor([False]),
            started=torch.tensor([False]),
            label_observed=torch.tensor([True]),
        )


@requires_torch
def test_match_contract_requires_sentinels_for_non_completed_games() -> None:
    import torch

    targets = MatchTargets(
        wdl_class=torch.tensor([2, -1, -1]),
        home_runs=torch.tensor([5, -1, -1]),
        away_runs=torch.tensor([2, -1, -1]),
        completed=torch.tensor([True, False, False]),
        result_observed=torch.tensor([True, True, False]),
    )

    assert torch.equal(
        targets.result_loss_mask,
        torch.tensor([True, False, False]),
    )
    assert targets.sample_count == 3
    assert targets.supervised_sample_count == 1

    with pytest.raises(ValueError, match="-1 result sentinels"):
        MatchTargets(
            wdl_class=torch.tensor([1]),
            home_runs=torch.tensor([0]),
            away_runs=torch.tensor([0]),
            completed=torch.tensor([False]),
            result_observed=torch.tensor([True]),
        )
    with pytest.raises(ValueError, match="result_observed=true"):
        MatchTargets(
            wdl_class=torch.tensor([2]),
            home_runs=torch.tensor([3]),
            away_runs=torch.tensor([1]),
            completed=torch.tensor([True]),
            result_observed=torch.tensor([False]),
        )


def _build_model() -> Any:
    import torch
    from torch import nn

    from cpv26.models import (
        DirectPlayerGameHead,
        DirectRunDistributionHead,
        PlateAppearanceInteractionDecoder,
        WDLHead,
    )

    class SharedBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(3, 4)

        def forward(self, features: Any) -> Any:
            return torch.tanh(self.projection(features))

    class PAAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(4, 4)

        def forward(self, shared: Any) -> tuple[Any, Any]:
            projected = self.projection(shared)
            return projected[:, 0], projected[:, 1]

    class LiveHitAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(4, 4)

        def forward(self, shared: Any) -> Any:
            return self.projection(shared)

    class MatchAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(4, 4)

        def forward(self, shared: Any) -> dict[str, Any]:
            projected = self.projection(shared)
            return {
                "home_embedding": projected[:, 0],
                "away_embedding": projected[:, 1],
            }

    return TaskSeparatedModel(
        backbone=SharedBackbone(),
        pa_adapter=PAAdapter(),
        live_hit_adapter=LiveHitAdapter(),
        match_adapter=MatchAdapter(),
        pa_head=PlateAppearanceInteractionDecoder(4, hidden_dim=8, dropout=0.0),
        live_hit_head=DirectPlayerGameHead(
            4,
            hidden_dim=8,
            max_plate_appearances=4,
            max_hits=3,
            dropout=0.0,
        ),
        wdl_head=WDLHead(4, hidden_dim=8, dropout=0.0),
        run_head=DirectRunDistributionHead(4, hidden_dim=8, dropout=0.0),
    )


def _task_batches() -> dict[str, Any]:
    import torch

    return {
        PA_TASK: PATaskBatch(
            backbone_inputs={"features": torch.randn((2, 2, 3))},
            targets=PATargets(torch.tensor([0, 3])),
        ),
        LIVE_HIT_TASK: LiveHitTaskBatch(
            backbone_inputs={"features": torch.randn((2, 3))},
            targets=LiveHitTargets(
                appeared=torch.tensor([False, True]),
                plate_appearances=torch.tensor([0, 4]),
                hits=torch.tensor([0, 2]),
                game_played=torch.tensor([True, True]),
                started=torch.tensor([False, True]),
                label_observed=torch.tensor([True, True]),
            ),
        ),
        MATCH_TASK: MatchTaskBatch(
            backbone_inputs={"features": torch.randn((2, 2, 3))},
            targets=MatchTargets(
                wdl_class=torch.tensor([2, 0]),
                home_runs=torch.tensor([5, 1]),
                away_runs=torch.tensor([2, 3]),
                completed=torch.tensor([True, True]),
                result_observed=torch.tensor([True, True]),
            ),
        ),
    }


@requires_torch
def test_live_hit_loss_uses_only_observed_played_game_rows() -> None:
    import torch

    torch.manual_seed(101)
    model = _build_model()
    model.eval()
    features = torch.randn((4, 3))
    mixed = LiveHitTaskBatch(
        backbone_inputs={"features": features},
        targets=LiveHitTargets(
            appeared=torch.tensor([True, False, False, False]),
            plate_appearances=torch.tensor([4, 0, 0, 0]),
            hits=torch.tensor([2, 0, 0, 0]),
            game_played=torch.tensor([True, False, False, True]),
            started=torch.tensor([True, False, False, False]),
            label_observed=torch.tensor([True, True, False, True]),
        ),
    )
    selected = LiveHitTaskBatch(
        backbone_inputs={"features": features[[0, 3]]},
        targets=LiveHitTargets(
            appeared=torch.tensor([True, False]),
            plate_appearances=torch.tensor([4, 0]),
            hits=torch.tensor([2, 0]),
            game_played=torch.tensor([True, True]),
            started=torch.tensor([True, False]),
            label_observed=torch.tensor([True, True]),
        ),
    )
    composer = MultiTaskLossComposer()

    mixed_loss = composer.live_hit_loss(model, mixed)
    selected_loss = composer.live_hit_loss(model, selected)

    assert mixed_loss.sample_count == 2
    assert torch.allclose(mixed_loss.total, selected_loss.total)
    assert torch.allclose(
        mixed_loss.components["joint_nll"],
        selected_loss.components["joint_nll"],
    )

    no_supervision = LiveHitTaskBatch(
        backbone_inputs={"features": features[:2]},
        targets=LiveHitTargets(
            appeared=torch.tensor([False, False]),
            plate_appearances=torch.tensor([0, 0]),
            hits=torch.tensor([0, 0]),
            game_played=torch.tensor([False, False]),
            started=torch.tensor([False, False]),
            label_observed=torch.tensor([True, False]),
        ),
    )
    with pytest.raises(ValueError, match="no observed played-game labels"):
        composer.live_hit_loss(model, no_supervision)


@requires_torch
def test_match_loss_excludes_cancelled_and_unresolved_rows() -> None:
    import torch

    torch.manual_seed(103)
    model = _build_model()
    model.eval()
    features = torch.randn((3, 2, 3))
    mixed = MatchTaskBatch(
        backbone_inputs={"features": features},
        targets=MatchTargets(
            wdl_class=torch.tensor([2, -1, -1]),
            home_runs=torch.tensor([5, -1, -1]),
            away_runs=torch.tensor([2, -1, -1]),
            completed=torch.tensor([True, False, False]),
            result_observed=torch.tensor([True, True, False]),
        ),
    )
    selected = MatchTaskBatch(
        backbone_inputs={"features": features[:1]},
        targets=MatchTargets(
            wdl_class=torch.tensor([2]),
            home_runs=torch.tensor([5]),
            away_runs=torch.tensor([2]),
            completed=torch.tensor([True]),
            result_observed=torch.tensor([True]),
        ),
    )
    composer = MultiTaskLossComposer()

    mixed_loss = composer.match_loss(model, mixed)
    selected_loss = composer.match_loss(model, selected)

    assert mixed_loss.sample_count == 1
    assert torch.allclose(mixed_loss.total, selected_loss.total)
    assert torch.allclose(
        mixed_loss.components["wdl_cross_entropy"],
        selected_loss.components["wdl_cross_entropy"],
    )
    assert torch.allclose(
        mixed_loss.components["run_nll"],
        selected_loss.components["run_nll"],
    )

    no_supervision = MatchTaskBatch(
        backbone_inputs={"features": features[1:]},
        targets=MatchTargets(
            wdl_class=torch.tensor([-1, -1]),
            home_runs=torch.tensor([-1, -1]),
            away_runs=torch.tensor([-1, -1]),
            completed=torch.tensor([False, False]),
            result_observed=torch.tensor([True, False]),
        ),
    )
    with pytest.raises(ValueError, match="no completed observed results"):
        composer.match_loss(model, no_supervision)


@requires_torch
@pytest.mark.parametrize("task", [PA_TASK, LIVE_HIT_TASK, MATCH_TASK])
def test_each_task_loss_reaches_shared_backbone_and_only_its_heads(task: str) -> None:
    import torch

    torch.manual_seed(19)
    model = _build_model()
    batch = _task_batches()[task]
    composer = MultiTaskLossComposer()
    model.zero_grad(set_to_none=True)

    loss = composer(model, task, batch)
    loss.total.backward()

    assert torch.isfinite(loss.total)
    assert model.backbone.projection.weight.grad is not None
    assert torch.isfinite(model.backbone.projection.weight.grad).all()
    head_has_gradient = {
        PA_TASK: next(model.pa_head.parameters()).grad is not None,
        LIVE_HIT_TASK: next(model.live_hit_head.parameters()).grad is not None,
        MATCH_TASK: (
            next(model.wdl_head.parameters()).grad is not None
            and next(model.run_head.parameters()).grad is not None
        ),
    }
    assert head_has_gradient[task]
    if task != PA_TASK:
        assert next(model.pa_head.parameters()).grad is None
    if task != LIVE_HIT_TASK:
        assert next(model.live_hit_head.parameters()).grad is None
    if task != MATCH_TASK:
        assert next(model.wdl_head.parameters()).grad is None
        assert next(model.run_head.parameters()).grad is None


@requires_torch
def test_alternating_trainer_uses_distinct_loaders_and_restores_checkpoint() -> None:
    import torch

    torch.manual_seed(23)
    model = _build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lineage = CheckpointLineage(
        feature_version="features-v1",
        route_version="routes-v1",
        label_schema_version="labels-v1",
        model_version="task-model-v1",
    )
    trainer = AlternatingMultiTaskTrainer(
        model=model,
        optimizer=optimizer,
        seed=41,
        checkpoint_lineage=lineage,
    )
    batches = _task_batches()
    records = trainer.train_epoch(
        {
            PA_TASK: [batches[PA_TASK], batches[PA_TASK]],
            LIVE_HIT_TASK: [batches[LIVE_HIT_TASK]],
            MATCH_TASK: [batches[MATCH_TASK]],
        }
    )

    assert [record.task for record in records] == [
        PA_TASK,
        LIVE_HIT_TASK,
        MATCH_TASK,
        PA_TASK,
    ]
    assert trainer.global_step == 4
    assert trainer.task_steps == {PA_TASK: 2, LIVE_HIT_TASK: 1, MATCH_TASK: 1}
    assert all(math.isfinite(record.loss) for record in records)

    checkpoint = trainer.checkpoint_state()
    restored_model = _build_model()
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
    restored = AlternatingMultiTaskTrainer(
        model=restored_model,
        optimizer=restored_optimizer,
        checkpoint_lineage=lineage,
    )
    restored.load_checkpoint_state(checkpoint)

    assert restored.epoch == 1
    assert restored.global_step == 4
    assert restored.task_steps == trainer.task_steps
    for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(expected, actual)

    incompatible_model = _build_model()
    incompatible = AlternatingMultiTaskTrainer(
        model=incompatible_model,
        optimizer=torch.optim.Adam(incompatible_model.parameters(), lr=1e-3),
        checkpoint_lineage=CheckpointLineage(
            feature_version="features-v2",
            route_version="routes-v1",
            label_schema_version="labels-v1",
            model_version="task-model-v1",
        ),
    )
    with pytest.raises(ValueError, match="checkpoint lineage does not match"):
        incompatible.load_checkpoint_state(checkpoint)
