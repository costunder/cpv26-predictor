"""CUDA-first, resumable training and held-out evaluation of real KBO RelGNNs."""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import date, datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from cpv26.data.kbo_graph_dataset import GraphDay, KBOGraphDataset
from cpv26.data.kbo_playbyplay import sha256_file
from cpv26.evaluation import evaluate_probabilities
from cpv26.models._torch import require_torch
from cpv26.models.kbo_relgnn import (
    KBORelGNNConfig,
    KBORelGNNModel,
    collate_kbo_day_graphs,
    kbo_multitask_loss,
)
from cpv26.simulation.adapter import NEURAL_PA_OUTCOMES

CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class KBOTrainingConfig:
    device: str = "cuda:0"
    epochs: int = 30
    batch_days: int = 2
    hidden_dim: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    amp: str = "auto"
    workers: int = 2
    accumulate_steps: int = 1
    gradient_clip: float = 1.0
    max_pa_per_day: int = 128
    max_edges_per_route_per_day: int = 20000
    patience: int = 6
    seed: int = 2026
    match_weight: float = 1.0
    live_hit_weight: float = 1.0
    pa_weight: float = 0.2
    run_weight: float = 0.1
    max_days_per_split: int | None = None
    train_seasons: tuple[int, ...] = (2023,)
    validation_season: int = 2024
    test_season: int = 2025

    def __post_init__(self) -> None:
        positive = (
            "epochs",
            "batch_days",
            "hidden_dim",
            "layers",
            "heads",
            "accumulate_steps",
            "max_pa_per_day",
            "max_edges_per_route_per_day",
        )
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "match_weight", "live_hit_weight", "pa_weight", "run_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.match_weight <= 0 or self.live_hit_weight <= 0:
            raise ValueError("both match and Live Hit tasks must have positive weights")
        if min(self.workers, self.patience, self.seed) < 0:
            raise ValueError("workers, patience and seed must be non-negative")
        if self.max_days_per_split is not None and self.max_days_per_split < 1:
            raise ValueError("max_days_per_split must be positive when supplied")
        if self.amp not in {"auto", "off", "fp16", "bf16"}:
            raise ValueError("amp must be auto, off, fp16, or bf16")
        if not self.train_seasons or not (
            max(self.train_seasons) < self.validation_season < self.test_season
        ):
            raise ValueError("training seasons must precede validation and held-out test")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KBOTrainingConfig:
        options = dict(value)
        options["train_seasons"] = tuple(options.get("train_seasons", (2023,)))
        return cls(**options)


def _device_and_precision(requested: str, amp: str) -> tuple[Any, Any | None, dict[str, Any]]:
    torch, _ = require_torch()
    if amp not in {"auto", "off", "fp16", "bf16"}:
        raise ValueError("amp must be auto, off, fp16, or bf16")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("this training path supports an explicit cuda device or CPU validation")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Install a compatible CUDA PyTorch build on the Linux "
                "GPU server and run `cpv26 gpu-check`. CPU fallback is disabled."
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is not visible to this process")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        if amp == "auto":
            amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        if amp == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("this GPU does not support bf16; use --amp fp16 or --amp off")
        details: dict[str, Any] = {
            "device": str(device),
            "gpu_name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "compiled_cuda_architectures": list(torch.cuda.get_arch_list()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    else:
        if amp not in {"auto", "off"}:
            raise ValueError("CPU validation uses --amp off; fp16/bf16 are CUDA options here")
        amp = "off"
        details = {"device": "cpu", "gpu_name": None, "total_memory_bytes": None}
    dtype = {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}[amp]
    details.update(
        torch_version=str(torch.__version__),
        cuda_runtime=torch.version.cuda,
        precision=amp,
        seeded_but_cuda_atomics_may_be_nondeterministic=True,
    )
    return device, dtype, details


def check_gpu(device: str = "cuda:0", *, amp: str = "auto") -> dict[str, Any]:
    """Exercise real matmul/autocast/backward kernels, not just a CUDA availability flag."""
    torch, _ = require_torch()
    selected, dtype, details = _device_and_precision(device, amp)
    x = torch.randn(128, 64, device=selected, requires_grad=True)
    weight = torch.randn(64, 32, device=selected, requires_grad=True)
    with torch.autocast(selected.type, enabled=dtype is not None, dtype=dtype):
        loss = (x @ weight).float().square().mean()
    loss.backward()
    if not bool(torch.isfinite(x.grad).all()) or not bool(torch.isfinite(weight.grad).all()):
        raise FloatingPointError("GPU kernel check produced non-finite gradients")
    if selected.type == "cuda":
        torch.cuda.synchronize(selected)
        free, total = torch.cuda.mem_get_info(selected)
        details.update(free_memory_bytes=free, total_memory_bytes=total)
    details["forward_backward_verified"] = True
    return details


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        partial_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        partial_path.replace(path)
    finally:
        partial_path.unlink(missing_ok=True)


def _atomic_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    torch, _ = require_torch()
    partial_path = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        torch.save(dict(state), partial_path)
        partial_path.replace(path)
    finally:
        partial_path.unlink(missing_ok=True)


def _read_checkpoint(path: Path) -> dict[str, Any]:
    torch, _ = require_torch()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported KBO RelGNN checkpoint")
    return state


def _move(value: Any, device: Any) -> Any:
    torch, _ = require_torch()
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value, **{item.name: _move(getattr(value, item.name), device) for item in fields(value)}
        )
    return value


class _DayDataset:
    def __init__(self, directory: Path, days: Sequence[date]) -> None:
        self.directory = directory
        self.selected_days = tuple(days)
        self._dataset: KBOGraphDataset | None = None

    def __len__(self) -> int:
        return len(self.selected_days)

    def __getitem__(self, index: int) -> GraphDay:
        if self._dataset is None:
            self._dataset = KBOGraphDataset(self.directory)
        return self._dataset.load_day(self.selected_days[index])


def _loader(
    directory: Path,
    days: Sequence[date],
    config: KBOTrainingConfig,
    *,
    epoch: int,
    training: bool,
) -> Any:
    torch, _ = require_torch()
    generator = torch.Generator().manual_seed(config.seed + epoch)
    return torch.utils.data.DataLoader(
        _DayDataset(directory, days),
        batch_size=config.batch_days,
        shuffle=training,
        num_workers=config.workers,
        pin_memory=config.device.startswith("cuda"),
        generator=generator,
        collate_fn=partial(
            collate_kbo_day_graphs,
            device="cpu",
            max_pa_per_day=config.max_pa_per_day if training else None,
            max_edges_per_route_per_day=config.max_edges_per_route_per_day,
            seed=config.seed + epoch if training else config.seed,
        ),
    )


def _split_days(
    dataset: KBOGraphDataset,
    config: KBOTrainingConfig,
) -> dict[str, tuple[date, ...]]:
    selected: dict[str, tuple[date, ...]] = {}
    for name, years in (
        ("train", config.train_seasons),
        ("validation", (config.validation_season,)),
        ("test", (config.test_season,)),
    ):
        days = tuple(day for day in dataset.days() if day.year in years)
        if config.max_days_per_split is not None and len(days) > config.max_days_per_split:
            indices = np.linspace(0, len(days) - 1, config.max_days_per_split, dtype=int)
            days = tuple(days[int(index)] for index in indices)
        selected[name] = days
    if not selected["train"] or not selected["validation"]:
        raise ValueError("graph dataset needs non-empty 2023 training and 2024 validation dates")
    return selected


def _losses(outputs: Mapping[str, Any], batch: Mapping[str, Any], config: KBOTrainingConfig) -> Any:
    return kbo_multitask_loss(
        outputs,
        batch,
        match_weight=config.match_weight,
        live_hit_weight=config.live_hit_weight,
        pa_weight=config.pa_weight,
        run_weight=config.run_weight,
    )


def _counts(batch: Mapping[str, Any]) -> dict[str, int]:
    return {
        "match": int(batch["match_targets"].numel()),
        "live_hit": int(batch["live_hit_pa"].numel()),
        "pa": int(batch["pa_targets"].numel()),
        "run": int(batch["match_targets"].numel()),
    }


def _model_config(dataset: KBOGraphDataset, config: KBOTrainingConfig) -> KBORelGNNConfig:
    manifest = dataset.manifest
    return KBORelGNNConfig(
        node_feature_dims=dict(manifest["node_feature_dims"]),
        role_feature_dims=dict(manifest["player_role_feature_dims"]),
        route_feature_dims=dict(manifest["route_feature_dims"]),
        hidden_dim=config.hidden_dim,
        num_layers=config.layers,
        num_attention_heads=config.heads,
        dropout=config.dropout,
        include_run_head=config.run_weight > 0,
    )


def _runtime_memory(device: Any) -> dict[str, int]:
    torch, _ = require_torch()
    if device.type != "cuda":
        return {}
    return {
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _probability_report(targets: np.ndarray[Any, Any], values: np.ndarray[Any, Any]) -> Any:
    if targets.size == 0:
        return None
    return {
        "samples": int(targets.size),
        **asdict(evaluate_probabilities(targets, values)),
        "accuracy": float(np.mean(np.argmax(values, axis=1) == targets)),
    }


def _float64_probabilities(value: Any) -> np.ndarray[Any, Any]:
    """Remove float32 summation roundoff without accepting malformed distributions."""
    values = np.asarray(value, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < -2e-6) or np.any(values > 1 + 2e-6):
        raise FloatingPointError("invalid model probabilities")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=2e-6, atol=2e-6):
        raise FloatingPointError("model probabilities are not normalized")
    values = np.clip(values, 0.0, 1.0)
    return np.asarray(values / values.sum(axis=1, keepdims=True), dtype=np.float64)


def _evaluate_model(
    model: Any,
    loader: Any,
    config: KBOTrainingConfig,
    device: Any,
    dtype: Any,
    *,
    collect_predictions: bool = False,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    torch, _ = require_torch()
    model.eval()
    sums = dict.fromkeys(("match", "live_hit", "pa", "run"), 0.0)
    counts = dict.fromkeys(sums, 0)
    probabilities: dict[str, list[Any]] = {name: [] for name in ("match", "live_hit", "pa")}
    targets: dict[str, list[Any]] = {name: [] for name in probabilities}
    predictions: dict[str, list[dict[str, Any]]] = {name: [] for name in probabilities}
    hit_absolute_error = 0.0
    pa_absolute_error = 0.0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move(raw_batch, device)
            with torch.autocast(device.type, enabled=dtype is not None, dtype=dtype):
                outputs = model(batch)
                losses = _losses(outputs, batch, config)
            batch_counts = _counts(batch)
            for name, count in batch_counts.items():
                value = float(losses[f"{name}_loss"].detach().cpu())
                if not math.isfinite(value):
                    raise FloatingPointError(f"non-finite validation {name} loss")
                sums[name] += value * count
                counts[name] += count
            values = {
                "match": torch.softmax(outputs["match_logits"].float(), dim=-1),
                "live_hit": torch.stack(
                    (1 - outputs["live_hit_hit_probability"], outputs["live_hit_hit_probability"]),
                    dim=-1,
                ),
                "pa": torch.softmax(outputs["pa_logits"].float(), dim=-1),
            }
            labels = {
                "match": batch["match_targets"],
                "live_hit": (batch["live_hit_hits"] > 0).long(),
                "pa": batch["pa_targets"],
            }
            expected_hits = outputs["live_hit_expected_hits"].float().cpu().numpy()
            expected_pa = outputs["live_hit_expected_pa"].float().cpu().numpy()
            actual_hits = batch["live_hit_hits"].cpu().numpy()
            actual_pa = batch["live_hit_pa"].cpu().numpy()
            hit_absolute_error += float(np.abs(expected_hits - actual_hits).sum())
            pa_absolute_error += float(np.abs(expected_pa - actual_pa).sum())
            for name in probabilities:
                probability = _float64_probabilities(values[name].float().cpu().numpy())
                label = labels[name].cpu().numpy()
                probabilities[name].append(probability)
                targets[name].append(label)
                if collect_predictions:
                    query_ids = raw_batch[f"{name}_query_ids"]
                    for index, query_id in enumerate(query_ids):
                        row = {
                            "query_id": str(query_id),
                            "label": int(label[index]),
                            **{
                                f"probability_{column}": float(value)
                                for column, value in enumerate(probability[index])
                            },
                        }
                        if name == "live_hit":
                            row.update(
                                observed_hits=int(actual_hits[index]),
                                observed_pa=int(actual_pa[index]),
                                expected_hits_lower_bound=float(expected_hits[index]),
                                expected_pa_lower_bound=float(expected_pa[index]),
                            )
                        predictions[name].append(row)
    means = {name: sums[name] / max(1, counts[name]) for name in sums}
    result: dict[str, Any] = {
        "losses": means,
        "selection_loss": sum(means[name] * getattr(config, f"{name}_weight") for name in means),
    }
    for name in probabilities:
        if probabilities[name]:
            result[name] = _probability_report(
                np.concatenate(targets[name]),
                np.concatenate(probabilities[name]),
            )
        else:
            result[name] = None
    if result["live_hit"] is not None:
        result["live_hit"].update(
            conditional_on="at_least_one_observed_plate_appearance",
            joint_nll=means["live_hit"],
            expected_hits_lower_bound_mae=hit_absolute_error / counts["live_hit"],
            expected_pa_lower_bound_mae=pa_absolute_error / counts["live_hit"],
        )
    return result, predictions


def _checkpoint_state(
    *,
    model: Any,
    optimizer: Any,
    scaler: Any,
    dataset: KBOGraphDataset,
    dataset_directory: Path,
    config: KBOTrainingConfig,
    model_config: KBORelGNNConfig,
    epoch: int,
    global_step: int,
    best_score: float,
    best_epoch: int,
    stale_epochs: int,
    skipped_optimizer_steps: int,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    torch, _ = require_torch()
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "dataset_fingerprint": dataset.manifest["fingerprint"],
        "dataset_directory": str(dataset_directory),
        "model_config": model_config.to_dict(),
        "training_config": asdict(config),
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "history": list(history),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _resume_compatible(state: Mapping[str, Any], config: KBOTrainingConfig) -> None:
    previous = KBOTrainingConfig.from_dict(state["training_config"])
    mutable = {"epochs", "device", "workers", "batch_days", "amp", "accumulate_steps", "patience"}
    for key, value in asdict(previous).items():
        if key not in mutable and value != asdict(config)[key]:
            raise ValueError(f"resume configuration changes {key}; start a separate run instead")
    if config.epochs < int(state["epoch"]):
        raise ValueError("epochs is the total target and cannot precede the saved epoch")


def train_kbo_relgnn(
    dataset_directory: str | Path,
    run_directory: str | Path,
    *,
    config: KBOTrainingConfig | None = None,
    resume: str | Path | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train on 2023, select on 2024, and never inspect the 2025 holdout here."""
    torch, _ = require_torch()
    options = config or KBOTrainingConfig()
    device, dtype, runtime = _device_and_precision(options.device, options.amp)
    directory = Path(dataset_directory).expanduser().resolve()
    output = Path(run_directory).expanduser().resolve()
    dataset = KBOGraphDataset(directory)
    splits = _split_days(dataset, options)
    model_config = _model_config(dataset, options)
    torch.manual_seed(options.seed)
    random.seed(options.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(options.seed)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    model: Any = KBORelGNNModel(model_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    start_epoch, global_step, best_epoch, stale_epochs = 0, 0, 0, 0
    skipped_optimizer_steps = 0
    best_score = math.inf
    history: list[dict[str, Any]] = []
    if resume is not None:
        checkpoint = Path(resume).expanduser().resolve()
        if checkpoint.name != "last.pt" or checkpoint.parent != output:
            raise ValueError("resume must use last.pt inside the same run directory")
        state = _read_checkpoint(checkpoint)
        if state["dataset_fingerprint"] != dataset.manifest["fingerprint"]:
            raise ValueError("checkpoint dataset fingerprint differs from the graph dataset")
        if state["model_config"] != model_config.to_dict():
            raise ValueError("checkpoint model/feature/route configuration differs")
        _resume_compatible(state, options)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler.is_enabled() and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng_state"])
        random.setstate(state["python_rng_state"])
        if (
            device.type == "cuda"
            and state["cuda_rng_states"]
            and len(state["cuda_rng_states"]) == torch.cuda.device_count()
        ):
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        skipped_optimizer_steps = int(state.get("skipped_optimizer_steps", 0))
        best_epoch, best_score = int(state["best_epoch"]), float(state["best_score"])
        stale_epochs = int(state["stale_epochs"])
        history = list(state["history"])
        # The checkpoint is the committed epoch boundary. Discard partial log lines only.
        (output / "history.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in history),
            encoding="utf-8",
        )
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("run directory is not empty; use --resume or a new --run-dir")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            output / "config.json",
            {
                "dataset_directory": str(directory),
                "dataset_fingerprint": dataset.manifest["fingerprint"],
                "training": asdict(options),
                "model": model_config.to_dict(),
                "runtime": runtime,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    progress(
        f"RelGNN device={device} precision={runtime['precision']} "
        f"parameters={sum(p.numel() for p in model.parameters()):,}; "
        f"train_days={len(splits['train'])}, validation_days={len(splits['validation'])}; "
        "2025 test is not used during training"
    )
    if runtime.get("gpu_name"):
        progress(
            f"GPU: {runtime['gpu_name']}; VRAM={runtime['total_memory_bytes'] / 2**30:.1f} GiB"
        )
    for epoch in range(start_epoch, options.epochs):
        started = time.monotonic()
        model.train()
        loader = _loader(directory, splits["train"], options, epoch=epoch, training=True)
        optimizer.zero_grad(set_to_none=True)
        sums = dict.fromkeys(("match", "live_hit", "pa", "run"), 0.0)
        counts = dict.fromkeys(sums, 0)
        for index, raw_batch in enumerate(loader):
            batch = _move(raw_batch, device)
            with torch.autocast(device.type, enabled=dtype is not None, dtype=dtype):
                losses = _losses(model(batch), batch, options)
            if not bool(torch.isfinite(losses["loss"])):
                raise FloatingPointError("non-finite RelGNN training loss")
            group_start = (index // options.accumulate_steps) * options.accumulate_steps
            group_size = min(options.accumulate_steps, len(loader) - group_start)
            scaler.scale(losses["loss"] / group_size).backward()
            batch_counts = _counts(batch)
            for name, count in batch_counts.items():
                sums[name] += float(losses[f"{name}_loss"].detach().cpu()) * count
                counts[name] += count
            if (index + 1) % options.accumulate_steps == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options.gradient_clip)
                if not scaler.is_enabled() and not bool(torch.isfinite(norm)):
                    raise FloatingPointError("non-finite RelGNN gradients")
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() < previous_scale:
                    skipped_optimizer_steps += 1
                else:
                    global_step += 1
            if (index + 1) % 10 == 0 or index + 1 == len(loader):
                progress(
                    f"epoch {epoch + 1}/{options.epochs} batch {index + 1}/{len(loader)} "
                    f"loss={float(losses['loss'].detach().cpu()):.4f}"
                )
        validation, _ = _evaluate_model(
            model,
            _loader(directory, splits["validation"], options, epoch=epoch, training=False),
            options,
            device,
            dtype,
        )
        score = float(validation["selection_loss"])
        improved = score < best_score
        if improved:
            best_score, best_epoch, stale_epochs = score, epoch + 1, 0
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "skipped_optimizer_steps": skipped_optimizer_steps,
            "train_losses": {name: sums[name] / max(1, counts[name]) for name in sums},
            "training_samples": counts,
            "validation": validation,
            "elapsed_seconds": time.monotonic() - started,
            "best_epoch": best_epoch,
            "batch_days": options.batch_days,
            "accumulate_steps": options.accumulate_steps,
            **_runtime_memory(device),
        }
        history.append(record)
        state = _checkpoint_state(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            dataset=dataset,
            dataset_directory=directory,
            config=options,
            model_config=model_config,
            epoch=epoch + 1,
            global_step=global_step,
            best_score=best_score,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            skipped_optimizer_steps=skipped_optimizer_steps,
            history=history,
        )
        if improved:
            _atomic_checkpoint(output / "best.pt", state)
        _atomic_checkpoint(output / "last.pt", state)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        progress(
            f"epoch {epoch + 1}: validation_loss={score:.4f}; best_epoch={best_epoch}; "
            f"elapsed={record['elapsed_seconds']:.1f}s; checkpoint={output / 'last.pt'}"
        )
        if options.patience and stale_epochs >= options.patience:
            progress(f"Early stopping after {stale_epochs} validation epochs without improvement.")
            break
    report = {
        "status": "completed",
        "model": "role_aware_composite_relgnn",
        "runtime": runtime,
        "configuration": asdict(options),
        "model_config": model_config.to_dict(),
        "dataset_fingerprint": dataset.manifest["fingerprint"],
        "training_seasons": list(options.train_seasons),
        "validation_season": options.validation_season,
        "held_out_test_season": options.test_season,
        "test_used_during_training": False,
        "smoke_test_only": options.max_days_per_split is not None,
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_score,
        "optimizer_steps": global_step,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "best_checkpoint": str(output / "best.pt"),
        "last_checkpoint": str(output / "last.pt"),
        "best_checkpoint_sha256": sha256_file(output / "best.pt"),
        "last_checkpoint_sha256": sha256_file(output / "last.pt"),
        "live_hit_population": "observed_PA_at_least_one; not unconditional V26 candidates",
        "history": history,
        **_runtime_memory(device),
    }
    _atomic_json(output / "training_report.json", report)
    return report


def _write_prediction_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import duckdb

    if not rows:
        return
    names = tuple(rows[0])
    types = [
        "VARCHAR"
        if isinstance(rows[0][name], str)
        else "BIGINT"
        if isinstance(rows[0][name], int)
        else "DOUBLE"
        for name in names
    ]
    expressions = ", ".join(
        f"unnest(?::{kind}[]) AS {name}" for name, kind in zip(names, types, strict=True)
    )
    with duckdb.connect() as connection:
        connection.execute(
            f"CREATE TABLE predictions AS SELECT {expressions}",
            [[row[name] for row in rows] for name in names],
        )
        connection.execute("COPY predictions TO ? (FORMAT PARQUET)", [str(path)])


def evaluate_kbo_relgnn(
    checkpoint: str | Path,
    *,
    dataset_directory: str | Path | None = None,
    split: str = "test",
    device: str = "cuda:0",
    amp: str = "auto",
    batch_days: int = 2,
    workers: int = 2,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Reload an explicit checkpoint and write immutable, per-query held-out predictions."""
    torch, _ = require_torch()
    path = Path(checkpoint).expanduser().resolve()
    state = _read_checkpoint(path)
    options = replace(
        KBOTrainingConfig.from_dict(state["training_config"]),
        device=device,
        amp=amp,
        batch_days=batch_days,
        workers=workers,
    )
    selected, dtype, runtime = _device_and_precision(device, amp)
    directory = Path(dataset_directory or state["dataset_directory"]).expanduser().resolve()
    dataset = KBOGraphDataset(directory)
    if dataset.manifest["fingerprint"] != state["dataset_fingerprint"]:
        raise ValueError("evaluation graph dataset differs from the checkpoint fingerprint")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    days = _split_days(dataset, options)[split]
    if not days:
        raise ValueError(f"no dates available for the requested {split} split")
    model: Any = KBORelGNNModel(KBORelGNNConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    model.to(selected)
    if selected.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    else:
        torch.cuda.reset_peak_memory_stats(selected)
    metrics, predictions = _evaluate_model(
        model,
        _loader(directory, days, options, epoch=0, training=False),
        options,
        selected,
        dtype,
        collect_predictions=True,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    output = (
        (
            Path(output_directory)
            if output_directory
            else path.parent / "evaluations" / f"{split}-{run_id}"
        )
        .expanduser()
        .resolve()
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("evaluation directory is not empty; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    for task, rows in predictions.items():
        if rows:
            target = output / f"{task}_predictions.parquet"
            _write_prediction_parquet(target, rows)
            artifacts[task] = {
                "path": str(target),
                "sha256": sha256_file(target),
                "rows": len(rows),
            }
    report = {
        "model": "role_aware_composite_relgnn",
        "split": split,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_epoch": state["epoch"],
        "dataset_fingerprint": dataset.manifest["fingerprint"],
        "date_start": min(days).isoformat(),
        "date_end": max(days).isoformat(),
        "days": len(days),
        "runtime": runtime,
        "metrics": metrics,
        "prediction_artifacts": artifacts,
        "output_directory": str(output),
        "smoke_test_only": options.max_days_per_split is not None,
        "class_order": {
            "match": ["L", "D", "W"],
            "live_hit": ["no_hit", "hit"],
            "pa": list(NEURAL_PA_OUTCOMES),
        },
        "limitations": [
            "Live Hit is conditional on at least one observed PA, not a full candidate pool.",
            "Joint PA/hit overflow buckets yield lower-bound expectations, not exact tail means.",
            "PA auxiliary queries contain pre-PA state; match and Live Hit remain pre-day tasks.",
            "Source publication times are reconstructed; this is a retrospective benchmark.",
        ],
        **_runtime_memory(selected),
    }
    _atomic_json(output / "metrics.json", report)
    return report
