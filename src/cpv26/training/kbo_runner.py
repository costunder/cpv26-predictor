"""CUDA-first, resumable training and held-out evaluation of real KBO RelGNNs."""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
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
    live_hit_observed_nll,
)
from cpv26.simulation.adapter import NEURAL_PA_OUTCOMES

from .batch_transfer import move_batch
from .optimizer_state import make_adamw, optimizer_parameter_names

CHECKPOINT_VERSION = 1
BOX_PITCH_TARGET_NAMES = (
    "batters_faced", "outs_recorded", "pitches_thrown", "at_bats", "hits_allowed",
    "home_runs_allowed", "walks_hbp", "strikeouts", "runs_allowed", "earned_runs",
)


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
    box_pa_weight: float = 0.2
    box_pitch_weight: float = 0.1
    selection_target: str = "auto"
    box_gradient_mode: str = "auto"
    max_days_per_split: int | None = None
    train_seasons: tuple[int, ...] = (2023,)
    validation_season: int = 2024
    test_season: int = 2025
    chronological: bool = False

    def __post_init__(self) -> None:
        positive = (
            "epochs",
            "batch_days",
            "hidden_dim",
            "layers",
            "heads",
            "accumulate_steps",
        )
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("max_pa_per_day", "max_edges_per_route_per_day"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative; 0 disables sampling")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "weight_decay", "match_weight", "live_hit_weight", "pa_weight", "run_weight",
            "box_pa_weight", "box_pitch_weight",
        ):
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
        if not isinstance(self.chronological, bool):
            raise ValueError("chronological must be a boolean")
        if self.selection_target not in {"auto", "match", "weighted"}:
            raise ValueError("selection_target must be auto, match, or weighted")
        if self.box_gradient_mode not in {"auto", "shared", "head_only"}:
            raise ValueError("box_gradient_mode must be auto, shared, or head_only")
        seasons = (*self.train_seasons, self.validation_season, self.test_season)
        if any(
            isinstance(year, bool) or not isinstance(year, int) or not 1 <= year <= 9999
            for year in seasons
        ):
            raise ValueError("season years must be integers between 1 and 9999")
        if tuple(sorted(set(self.train_seasons))) != self.train_seasons:
            raise ValueError("training seasons must be unique and increasing")
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
    return move_batch(value, device, packed=True)


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
    ordered_days = sorted(days) if config.chronological or not training else days
    return torch.utils.data.DataLoader(
        _DayDataset(directory, ordered_days),
        batch_size=config.batch_days,
        shuffle=training and not config.chronological,
        num_workers=config.workers,
        # Workers start after model/CUDA initialization. Do not fork its
        # threaded runtime; keep this choice local to this DataLoader.
        multiprocessing_context="spawn" if config.workers > 0 else None,
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
    available_days = tuple(sorted(dataset.days()))
    available_years = {day.year for day in available_days}
    for name, years in (
        ("training", config.train_seasons),
        ("validation", (config.validation_season,)),
    ):
        missing = sorted(set(years) - available_years)
        if missing:
            raise ValueError(
                f"graph dataset has no dates for requested {name} seasons: "
                f"{', '.join(map(str, missing))}; "
                "import those seasons and build a new graph dataset"
            )
    selected: dict[str, tuple[date, ...]] = {}
    for name, years in (
        ("train", config.train_seasons),
        ("validation", (config.validation_season,)),
        ("test", (config.test_season,)),
    ):
        days = tuple(day for day in available_days if day.year in years)
        if config.max_days_per_split is not None and len(days) > config.max_days_per_split:
            indices = np.linspace(0, len(days) - 1, config.max_days_per_split, dtype=int)
            days = tuple(days[int(index)] for index in indices)
        selected[name] = days
    return selected


def _split_summary(
    dataset: KBOGraphDataset, splits: Mapping[str, Sequence[date]]
) -> dict[str, Any]:
    entries = {entry["day"]: entry for entry in dataset.manifest["days"]}
    summary: dict[str, Any] = {}
    for name, days in splits.items():
        rows = [entries[day.isoformat()] for day in days]
        summary[name] = {
            "seasons": sorted({day.year for day in days}),
            "days": len(days),
            "date_start": min(days).isoformat() if days else None,
            "date_end": max(days).isoformat() if days else None,
            "games": sum(row["games"] for row in rows),
            "live_hit_queries": sum(row["live_hit_queries"] for row in rows),
            "pa_queries": sum(row["pa_queries"] for row in rows),
        }
        for key in (
            "games_with_pa", "game_only_games", "observed_completed_pa",
            "box_batting_rows", "box_pitching_rows", "box_live_hit_queries",
            "box_live_hit_unknown_pa_queries",
        ):
            if any(key in row for row in rows):
                summary[name][key] = sum(row.get(key, 0) for row in rows)
        box_keys = (
            "box_pa_queries", "box_pa_outcomes", "box_pitch_queries", "box_pitch_observed_counts",
        )
        if int(dataset.manifest.get("dataset_version", 2)) >= 5:
            # v5 records the actual targets from both archive and modern PA.
            # A missing required field is corrupt metadata, not zero coverage.
            for key in box_keys:
                summary[name][key] = sum(row[key] for row in rows)
            summary[name]["box_coverage_source"] = "manifest_all_sources"
        else:
            # v4's flat counts were archive-only; modern derived targets were
            # present in NPZ but falsely reported as zero. Read legacy arrays
            # once for this report without rewriting caches/checkpoints.
            totals = dict.fromkeys(box_keys, 0)
            for day in days:
                graph = dataset.load_day(day)
                totals["box_pa_queries"] += len(graph.box_pa_counts)
                totals["box_pa_outcomes"] += int(graph.box_pa_counts.sum())
                totals["box_pitch_queries"] += len(graph.box_pitch_mask)
                totals["box_pitch_observed_counts"] += int(graph.box_pitch_mask.sum())
            summary[name].update(totals)
            summary[name]["box_coverage_source"] = "graph_arrays_legacy_manifest"
        if any("raw_archive_boxscore" in row for row in rows):
            summary[name]["raw_archive_boxscore"] = {
                key: sum(row.get("raw_archive_boxscore", {}).get(key, 0) for row in rows)
                for key in box_keys
            }
    return summary


def _losses(outputs: Mapping[str, Any], batch: Mapping[str, Any], config: KBOTrainingConfig) -> Any:
    return kbo_multitask_loss(
        outputs,
        batch,
        match_weight=config.match_weight,
        live_hit_weight=config.live_hit_weight,
        pa_weight=config.pa_weight,
        run_weight=config.run_weight,
        box_pa_weight=config.box_pa_weight,
        box_pitch_weight=config.box_pitch_weight,
    )


def _counts(batch: Mapping[str, Any], *, include_boxscore: bool = False) -> dict[str, int]:
    counts = {
        "match": int(batch["match_targets"].numel()),
        "live_hit": int(batch["live_hit_pa"].numel()),
        "pa": int(batch["pa_targets"].numel()),
        "run": int(batch["match_targets"].numel()),
    }
    if include_boxscore:
        counts["box_pa"] = int(batch["box_pa_counts"].sum().item())
        counts["box_pitch"] = int(batch["box_pitch_mask"].sum().item())
    return counts


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
        include_boxscore_heads=manifest["dataset_version"] >= 3,
        box_batting_feature_dim=manifest.get("boxscore_feature_dims", {}).get("batting", 19),
        box_pitching_feature_dim=manifest.get("boxscore_feature_dims", {}).get("pitching", 21),
        box_gradient_mode=_training_policies(config)["box_gradient_mode"],
    )


def _training_policies(config: KBOTrainingConfig) -> dict[str, str]:
    """Keep defaults stable; objective and gradient isolation are explicit controls."""
    return {
        "selection_target": "weighted" if config.selection_target == "auto"
        else config.selection_target,
        "box_gradient_mode": "shared" if config.box_gradient_mode == "auto"
        else config.box_gradient_mode,
    }


def _gradient_parameter_groups(model: Any) -> dict[str, list[Any]]:
    if model.config.include_boxscore_heads and model.config.box_gradient_mode == "head_only":
        groups: dict[str, list[Any]] = {"primary": [], "box_heads": []}
        for name, parameter in model.named_parameters():
            group = (
                "box_heads" if name.startswith(("box_pa_head.", "box_pitch_head.")) else "primary"
            )
            groups[group].append(parameter)
        return groups
    return {"shared": list(model.parameters())}


def _clip_gradient_norms(model: Any, max_norm: float) -> dict[str, Any]:
    """Detached auxiliary gradients must not rescale primary gradients via clipping."""
    torch, _ = require_torch()
    return {
        name: torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        for name, parameters in _gradient_parameter_groups(model).items()
    }


def _policy_report(config: KBOTrainingConfig, model_config: KBORelGNNConfig) -> dict[str, Any]:
    groups = "primary_and_box_heads_separately" if (
        model_config.include_boxscore_heads and model_config.box_gradient_mode == "head_only"
    ) else "all_parameters_together"
    return {
        "requested_selection_target": config.selection_target,
        "selection_target": _training_policies(config)["selection_target"],
        "requested_box_gradient_mode": config.box_gradient_mode,
        "box_gradient_mode": model_config.box_gradient_mode,
        "gradient_clipping": groups,
        "loss_weights": {
            name: getattr(config, f"{name}_weight")
            for name in ("match", "live_hit", "pa", "run", "box_pa", "box_pitch")
        },
        "scope": "box gradient isolation does not isolate the original match/live_hit/pa/run tasks",
        "fp16_overflow_policy": (
            "GradScaler-detected nonfinite gradients skip the whole optimizer step"
        ),
        "execution_optimizations": {
            "shared_bidirectional_route_context": True,
            "dtype_packed_cuda_input_transfer": True,
            "optimizer_algorithm_changed": False,
        },
    }


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
    batch_transform: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    diagnostics: Any | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    torch, _ = require_torch()
    model.eval()
    include_boxscore = model.config.include_boxscore_heads
    tasks = ("match", "live_hit", "pa", "run") + (
        ("box_pa", "box_pitch") if include_boxscore else ()
    )
    sums = dict.fromkeys(tasks, 0.0)
    counts = dict.fromkeys(sums, 0)
    probabilities: dict[str, list[Any]] = {name: [] for name in ("match", "live_hit", "pa")}
    targets: dict[str, list[Any]] = {name: [] for name in probabilities}
    predictions: dict[str, list[dict[str, Any]]] = {name: [] for name in probabilities}
    if include_boxscore:
        predictions.update(box_pa=[], box_pitch=[])
    hit_absolute_error = 0.0
    pa_absolute_error = 0.0
    known_pa_count = 0
    known_pa_nll = 0.0
    unknown_pa_nll = 0.0
    unknown_pa_minimum_overflow = 0
    box_pa_queries = 0
    box_pitch_queries = 0
    box_pitch_errors = np.zeros(10, dtype=np.float64)
    box_pitch_counts = np.zeros(10, dtype=np.int64)
    with torch.inference_mode():
        for raw_batch in loader:
            prepared_batch = (
                batch_transform(raw_batch) if batch_transform is not None else raw_batch
            )
            if diagnostics is not None:
                begin_batch = getattr(diagnostics, "begin_batch", None)
                if callable(begin_batch):
                    begin_batch(prepared_batch)
            batch = _move(prepared_batch, device)
            with torch.autocast(device.type, enabled=dtype is not None, dtype=dtype):
                outputs = (
                    model(batch, diagnostics_observer=diagnostics)
                    if diagnostics is not None
                    else model(batch)
                )
                losses = _losses(outputs, batch, config)
            batch_counts = _counts(batch, include_boxscore=include_boxscore)
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
            pa_minimum = batch["live_hit_pa_min"].cpu().numpy()
            known_pa = actual_pa >= 1
            unknown_pa_minimum_overflow += int(
                ((~known_pa) & (np.maximum(pa_minimum, actual_hits)
                               > outputs["live_hit_joint_logits"].shape[1])).sum()
            )
            known_pa_count += int(known_pa.sum())
            live_nll = live_hit_observed_nll(outputs["live_hit_joint_logits"], batch).cpu().numpy()
            known_pa_nll += float(live_nll[known_pa].sum())
            unknown_pa_nll += float(live_nll[~known_pa].sum())
            hit_absolute_error += float(np.abs(expected_hits - actual_hits).sum())
            pa_absolute_error += float(np.abs(expected_pa[known_pa] - actual_pa[known_pa]).sum())
            if include_boxscore:
                box_counts = batch["box_pa_counts"].cpu().numpy()
                box_probabilities = torch.softmax(outputs["box_pa_logits"].float(), dim=-1)
                box_probabilities_array = _float64_probabilities(box_probabilities.cpu().numpy())
                box_pa_queries += box_counts.shape[0]
                pitch_targets = batch["box_pitch_targets"].cpu().numpy()
                pitch_mask = batch["box_pitch_mask"].cpu().numpy()
                pitch_rates = outputs["box_pitch_rates"].float().cpu().numpy()
                box_pitch_queries += pitch_targets.shape[0]
                box_pitch_errors += (np.abs(pitch_rates - pitch_targets) * pitch_mask).sum(axis=0)
                box_pitch_counts += pitch_mask.sum(axis=0)
                if collect_predictions:
                    for index, query_id in enumerate(raw_batch["box_pa_query_ids"]):
                        predictions["box_pa"].append({
                            "query_id": str(query_id),
                            **{
                                f"observed_count_{column}": int(value)
                                for column, value in enumerate(box_counts[index])
                            },
                            **{
                                f"probability_{column}": float(value)
                                for column, value in enumerate(box_probabilities_array[index])
                            },
                        })
                    for index, query_id in enumerate(raw_batch["box_pitch_query_ids"]):
                        predictions["box_pitch"].append({
                            "query_id": str(query_id),
                            **{
                                f"observed_{name}": int(pitch_targets[index, column])
                                if pitch_mask[index, column] else None
                                for column, name in enumerate(BOX_PITCH_TARGET_NAMES)
                            },
                            **{
                                f"predicted_{name}": float(pitch_rates[index, column])
                                for column, name in enumerate(BOX_PITCH_TARGET_NAMES)
                            },
                        })
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
                                observed_pa=int(actual_pa[index]) if known_pa[index] else None,
                                expected_hits_lower_bound=float(expected_hits[index]),
                                expected_pa_lower_bound=float(expected_pa[index]),
                            )
                            if include_boxscore or not known_pa[index]:
                                row["observed_pa_lower_bound"] = int(pa_minimum[index])
                        predictions[name].append(row)
    means = {name: sums[name] / max(1, counts[name]) for name in sums}
    contributions = {name: means[name] * getattr(config, f"{name}_weight") for name in means}
    weighted_loss = sum(contributions.values())
    selection_target = _training_policies(config)["selection_target"]
    if selection_target == "match" and not counts["match"]:
        raise ValueError("match checkpoint selection requires observed match labels")
    result: dict[str, Any] = {
        "losses": means,
        "loss_sample_counts": counts,
        "weighted_loss_contributions": contributions,
        "weighted_multitask_loss": weighted_loss,
        "selection_target": selection_target,
        "selection_loss": means["match"] if selection_target == "match" else weighted_loss,
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
        unknown_pa_count = counts["live_hit"] - known_pa_count
        result["live_hit"].update(
            conditional_on=("verified_player_game_appearance" if include_boxscore
                            else "at_least_one_observed_plate_appearance"),
            joint_nll=known_pa_nll / known_pa_count if known_pa_count else None,
            expected_hits_lower_bound_mae=hit_absolute_error / counts["live_hit"],
            expected_pa_lower_bound_mae=(
                pa_absolute_error / known_pa_count if known_pa_count else None
            ),
        )
        if include_boxscore or unknown_pa_count:
            result["live_hit"].update(
                observed_nll=means["live_hit"],
                known_pa_samples=known_pa_count,
                unknown_pa_samples=unknown_pa_count,
                unknown_pa_minimum_overflow_samples=unknown_pa_minimum_overflow,
                partial_pa_nll=unknown_pa_nll / unknown_pa_count if unknown_pa_count else None,
                partial_pa_policy="sum joint mass over observed H and PA >= verified minimum",
                minimum_overflow_policy=(
                    "minima above PA overflow start select that whole overflow bucket; "
                    "no exact within-bucket PA is inferred"
                ),
            )
    if include_boxscore:
        result["box_pa"] = {
            "player_game_queries": box_pa_queries,
            "observed_outcomes": counts["box_pa"],
            "cross_entropy": means["box_pa"],
            "label_type": "verified player-game outcome histogram; not ordered PA events",
        } if counts["box_pa"] else None
        result["box_pitch"] = {
            "player_game_queries": box_pitch_queries,
            "observed_counts": counts["box_pitch"],
            "poisson_nll": means["box_pitch"],
            "per_field": {
                name: {
                    "samples": int(box_pitch_counts[index]),
                    "mae": float(box_pitch_errors[index] / box_pitch_counts[index])
                    if box_pitch_counts[index] else None,
                }
                for index, name in enumerate(BOX_PITCH_TARGET_NAMES)
            },
        } if counts["box_pitch"] else None
    # A legacy decoder can also score partial PA observations. If any such row
    # adds a minimum column, retain one consistent schema across all batches.
    live_predictions = predictions["live_hit"]
    if any("observed_pa_lower_bound" in row for row in live_predictions):
        for row in live_predictions:
            row.setdefault("observed_pa_lower_bound", row["observed_pa"])
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
        "optimizer_parameter_names": optimizer_parameter_names(model, optimizer),
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
    """Train on configured past seasons and select with a later validation season."""
    torch, _ = require_torch()
    options = config or KBOTrainingConfig()
    device, dtype, runtime = _device_and_precision(options.device, options.amp)
    directory = Path(dataset_directory).expanduser().resolve()
    output = Path(run_directory).expanduser().resolve()
    dataset = KBOGraphDataset(directory)
    splits = _split_days(dataset, options)
    split_summary = _split_summary(dataset, splits)
    training_order = "chronological" if options.chronological else "shuffled"
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
    optimizer = make_adamw(
        model,
        learning_rate=options.learning_rate,
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
        if KBORelGNNConfig(**state["model_config"]).to_dict() != model_config.to_dict():
            raise ValueError("checkpoint model/feature/route configuration differs")
        _resume_compatible(state, options)
        model.load_state_dict(state["model"])
        optimizer = make_adamw(
            model, learning_rate=options.learning_rate, weight_decay=options.weight_decay,
            checkpoint=state,
        )
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
                "training_order": training_order,
                "split_summary": split_summary,
                "model": model_config.to_dict(),
                "training_policies": _policy_report(options, model_config),
                "runtime": runtime,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    progress(
        f"RelGNN device={device} precision={runtime['precision']} "
        f"parameters={sum(p.numel() for p in model.parameters()):,}; "
        f"train_days={len(splits['train'])}, validation_days={len(splits['validation'])}; "
        f"{options.test_season} test is not used during training"
    )
    progress(
        f"Training seasons={','.join(map(str, options.train_seasons))}; "
        f"validation={options.validation_season}; test={options.test_season}; "
        f"order={training_order}; train_games={split_summary['train']['games']:,}"
    )
    policy = _policy_report(options, model_config)
    progress(
        f"Checkpoint selection={policy['selection_target']}; "
        f"box gradients={policy['box_gradient_mode']}; clipping={policy['gradient_clipping']}"
    )
    if options.chronological:
        progress(
            "Each epoch visits training dates oldest first; weights continue across seasons. "
            "Validation is held out, not an online predict-then-learn score."
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
        task_names = ("match", "live_hit", "pa", "run") + (
            ("box_pa", "box_pitch") if model_config.include_boxscore_heads else ()
        )
        sums = dict.fromkeys(task_names, 0.0)
        counts = dict.fromkeys(sums, 0)
        gradient_audit = {
            name: {"steps": 0, "clipped_steps": 0, "nonfinite_steps": 0,
                   "max_finite_preclip_norm": 0.0}
            for name in _gradient_parameter_groups(model)
        }
        for index, raw_batch in enumerate(loader):
            batch = _move(raw_batch, device)
            with torch.autocast(device.type, enabled=dtype is not None, dtype=dtype):
                losses = _losses(model(batch), batch, options)
            if not bool(torch.isfinite(losses["loss"])):
                raise FloatingPointError("non-finite RelGNN training loss")
            group_start = (index // options.accumulate_steps) * options.accumulate_steps
            group_size = min(options.accumulate_steps, len(loader) - group_start)
            scaler.scale(losses["loss"] / group_size).backward()
            batch_counts = _counts(batch, include_boxscore=model_config.include_boxscore_heads)
            for name, count in batch_counts.items():
                sums[name] += float(losses[f"{name}_loss"].detach().cpu()) * count
                counts[name] += count
            if (index + 1) % options.accumulate_steps == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                norms = _clip_gradient_norms(model, options.gradient_clip)
                finite_gradients = True
                for name, norm in norms.items():
                    value = float(norm.detach().cpu())
                    audit = gradient_audit[name]
                    audit["steps"] += 1
                    if math.isfinite(value):
                        audit["clipped_steps"] += int(value > options.gradient_clip)
                        audit["max_finite_preclip_norm"] = max(
                            audit["max_finite_preclip_norm"], value
                        )
                    else:
                        audit["nonfinite_steps"] += 1
                        finite_gradients = False
                if not scaler.is_enabled() and not finite_gradients:
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
            "gradient_audit": gradient_audit,
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
        "training_policies": policy,
        "dataset_fingerprint": dataset.manifest["fingerprint"],
        "training_seasons": list(options.train_seasons),
        "validation_season": options.validation_season,
        "held_out_test_season": options.test_season,
        "training_order": training_order,
        "evaluation_protocol": "fixed_chronological_holdout",
        "split_summary": split_summary,
        "test_used_during_training": False,
        "smoke_test_only": options.max_days_per_split is not None,
        "sampling_limits": {
            "training_pa_per_day": options.max_pa_per_day or None,
            "edges_per_route_per_day": options.max_edges_per_route_per_day or None,
            "evaluation_pa_per_day": None,
            "boxscore_queries": None,
            "zero_means_unlimited": True,
        },
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_score,
        "optimizer_steps": global_step,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "best_checkpoint": str(output / "best.pt"),
        "last_checkpoint": str(output / "last.pt"),
        "best_checkpoint_sha256": sha256_file(output / "best.pt"),
        "last_checkpoint_sha256": sha256_file(output / "last.pt"),
        "live_hit_population": (
            "verified_player_game_appearance; observed PA or historical box score; "
            "not unconditional V26 candidates"
            if model_config.include_boxscore_heads
            else "observed_PA_at_least_one; not unconditional V26 candidates"
        ),
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
    types = []
    for name in names:
        first_value = next((row[name] for row in rows if row[name] is not None), None)
        if name.startswith("observed_") and name != "observed_pa_known":
            kind = "BIGINT"
        elif isinstance(first_value, str):
            kind = "VARCHAR"
        elif isinstance(first_value, bool):
            kind = "BOOLEAN"
        elif isinstance(first_value, int):
            kind = "BIGINT"
        else:
            kind = "DOUBLE"
        types.append(kind)
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
        "training_seasons": list(options.train_seasons),
        "validation_season": options.validation_season,
        "held_out_test_season": options.test_season,
        "dataset_fingerprint": dataset.manifest["fingerprint"],
        "date_start": min(days).isoformat(),
        "date_end": max(days).isoformat(),
        "days": len(days),
        "runtime": runtime,
        "metrics": metrics,
        "training_policies": _policy_report(options, model.config),
        "prediction_artifacts": artifacts,
        "output_directory": str(output),
        "smoke_test_only": options.max_days_per_split is not None,
        "sampling_limits": {
            "evaluation_pa_per_day": None,
            "edges_per_route_per_day": options.max_edges_per_route_per_day or None,
            "boxscore_queries": None,
            "zero_means_unlimited": True,
        },
        "class_order": {
            "match": ["L", "D", "W"],
            "live_hit": ["no_hit", "hit"],
            "pa": list(NEURAL_PA_OUTCOMES),
            **({"box_pa": list(NEURAL_PA_OUTCOMES), "box_pitch": list(BOX_PITCH_TARGET_NAMES)}
               if model.config.include_boxscore_heads else {}),
        },
        "limitations": [
            ("Live Hit is conditional on verified PA appearance (PBP or box score), "
             "not a full candidate pool." if model.config.include_boxscore_heads else
             "Live Hit is conditional on at least one observed PA, not a full candidate pool."),
            "Joint PA/hit overflow buckets yield lower-bound expectations, not exact tail means.",
            "PA auxiliary queries contain pre-PA state; match and Live Hit remain pre-day tasks.",
            "Source publication times are reconstructed; this is a retrospective benchmark.",
            *([
                "Box-score outcome histograms are not reconstructed PA event sequences.",
                "Unknown PA labels marginalize joint mass above the verified PA minimum; "
                "PA MAE excludes them. Overflow buckets cannot resolve within-bucket minima.",
                "Pitching counts use masked auxiliary Poisson objectives, not a joint simulator.",
            ] if model.config.include_boxscore_heads else []),
        ],
        **_runtime_memory(selected),
    }
    _atomic_json(output / "metrics.json", report)
    return report
