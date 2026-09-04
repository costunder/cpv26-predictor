"""CUDA-first, resumable training and held-out evaluation of real KBO RelGNNs."""

from __future__ import annotations

import hashlib
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

from cpv26.data.kbo_dataset_loader import (
    KBOGraphDatasetLike,
    open_kbo_graph_dataset,
)
from cpv26.data.kbo_graph_dataset import GraphDay
from cpv26.data.kbo_graph_dataset import KBOGraphDataset as KBOGraphDataset
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

from .batch_transfer import move_batch, prefetch_batches
from .optimizer_state import make_adamw, optimizer_parameter_names
from .resource_telemetry import (
    host_resource_inventory,
    numeric_distribution,
    resource_snapshot,
    summarize_resource_interval,
    tensor_shape_manifest,
)

CHECKPOINT_VERSION = 1
GRAPH_CONTROL_PROTOCOL_VERSION = 1
ROUTE_MESSAGE_NORMALIZATIONS = ("none", "layer_norm")
ROUTE_SCHEDULE_PRESETS = ("full", "staged", "core", "node_only")
GRAPH_CONTROL_MODES = ("intact", "permuted_endpoints")
_FULL_ROUTE_GATE_KEYS = tuple(
    f"{route_name}__{direction}"
    for route_name in (
        "batter_pa_pitcher",
        "batter_participation_team",
        "pitcher_participation_team",
        "home_team_game_away_team",
    )
    for direction in ("forward", "reverse")
)
_CORE_ROUTE_GATE_KEYS = tuple(
    f"{route_name}__{direction}"
    for route_name in ("batter_pa_pitcher", "batter_participation_team")
    for direction in ("forward", "reverse")
)
_STAGED_FIRST_ROUTE_GATE_KEYS = (
    "batter_pa_pitcher__forward",
    "batter_pa_pitcher__reverse",
    "batter_participation_team__forward",
    "pitcher_participation_team__reverse",
    "home_team_game_away_team__forward",
    "home_team_game_away_team__reverse",
)
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
    max_pa_per_day: int = 0
    max_edges_per_route_per_day: int = 0
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
    route_message_normalization: str = "none"
    route_schedule: str = "full"
    route_edge_chunk_size: int = 0
    graph_control: str = "intact"
    graph_control_seed: int = 2026
    activation_checkpointing: bool = False
    compact_kbo_channels: bool = True

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
        if min(self.workers, self.patience, self.seed, self.graph_control_seed) < 0:
            raise ValueError("workers, patience and seeds must be non-negative")
        if self.max_days_per_split is not None and self.max_days_per_split < 1:
            raise ValueError("max_days_per_split must be positive when supplied")
        if self.amp not in {"auto", "off", "fp16", "bf16"}:
            raise ValueError("amp must be auto, off, fp16, or bf16")
        if not isinstance(self.chronological, bool):
            raise ValueError("chronological must be a boolean")
        if not isinstance(self.activation_checkpointing, bool):
            raise ValueError("activation_checkpointing must be a boolean")
        if not isinstance(self.compact_kbo_channels, bool):
            raise ValueError("compact_kbo_channels must be a boolean")
        if self.selection_target not in {"auto", "match", "weighted"}:
            raise ValueError("selection_target must be auto, match, or weighted")
        if self.box_gradient_mode not in {"auto", "shared", "head_only"}:
            raise ValueError("box_gradient_mode must be auto, shared, or head_only")
        if self.route_message_normalization not in ROUTE_MESSAGE_NORMALIZATIONS:
            raise ValueError("route_message_normalization must be none or layer_norm")
        if self.route_schedule not in ROUTE_SCHEDULE_PRESETS:
            raise ValueError("route_schedule must be full, staged, core, or node_only")
        if (
            isinstance(self.route_edge_chunk_size, bool)
            or not isinstance(self.route_edge_chunk_size, int)
            or self.route_edge_chunk_size < 0
        ):
            raise ValueError("route_edge_chunk_size must be a non-negative integer")
        if self.graph_control not in GRAPH_CONTROL_MODES:
            raise ValueError("graph_control must be intact or permuted_endpoints")
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
        options.setdefault("route_message_normalization", "none")
        options.setdefault("route_schedule", "full")
        options.setdefault("route_edge_chunk_size", 0)
        options.setdefault("graph_control", "intact")
        options.setdefault("graph_control_seed", 2026)
        options.setdefault("activation_checkpointing", False)
        options.setdefault("compact_kbo_channels", False)
        return cls(**options)


@dataclass(frozen=True, slots=True)
class _TemporalExecution:
    """Immutable selected CUDA plan shared by both temporal variants."""

    report_path: Path
    report_sha256: str
    plan_fingerprint: str
    variant: str
    rows: tuple[Mapping[str, Any], ...]
    loader_workers: int
    loader_prefetch_factor: int | None
    loader_persistent_workers: bool
    preflight_resources: Mapping[str, Any]
    loader_autotune: Mapping[str, Any]

    def split_rows(self, split: str) -> tuple[Mapping[str, Any], ...]:
        return tuple(row for row in self.rows if row["split"] == split)

    def lineage(self) -> dict[str, Any]:
        return {
            "report_path": str(self.report_path),
            "report_sha256": self.report_sha256,
            "plan_fingerprint": self.plan_fingerprint,
            "variant": self.variant,
            "train_batch_count": len(self.split_rows("train")),
            "validation_batch_count": len(self.split_rows("validation")),
            "loader_runtime": {
                "workers": self.loader_workers,
                "prefetch_factor": self.loader_prefetch_factor,
                "persistent_workers": self.loader_persistent_workers,
                "packed_transfers": True,
            },
            "preflight_resource_measurements": dict(self.preflight_resources),
            "loader_autotune": dict(self.loader_autotune),
        }


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


def _model_state_sha256(model: Any) -> str:
    """Hash a model state independently of torch.save container metadata."""

    torch, _ = require_torch()
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not torch.is_tensor(value):
            raise TypeError(f"model state {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _parameter_state_sha256(
    model: Any, parameter_names: Sequence[str] | None = None
) -> str:
    """Hash named parameter values, optionally over a declared shared subset."""

    torch, _ = require_torch()
    parameters = dict(model.named_parameters())
    selected = tuple(sorted(parameters if parameter_names is None else parameter_names))
    missing = sorted(set(selected).difference(parameters))
    if missing:
        raise ValueError(f"shared initialization names are absent from model: {missing}")
    digest = hashlib.sha256()
    for name in selected:
        value = parameters[name]
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _move(value: Any, device: Any) -> Any:
    return move_batch(value, device, packed=True)


class _DayDataset:
    def __init__(
        self,
        directory: Path,
        days: Sequence[date],
        *,
        expected_sample_fingerprints: Mapping[date, str] | None = None,
        label_year_ceiling: int | None = None,
    ) -> None:
        self.directory = directory
        self.selected_days = tuple(days)
        self.expected_sample_fingerprints = (
            None
            if expected_sample_fingerprints is None
            else dict(expected_sample_fingerprints)
        )
        self.label_year_ceiling = label_year_ceiling
        self._dataset: KBOGraphDatasetLike | None = None

    def __len__(self) -> int:
        return len(self.selected_days)

    def __getitem__(self, index: int) -> GraphDay:
        if self._dataset is None:
            self._dataset = open_kbo_graph_dataset(
                self.directory,
                label_year_ceiling=self.label_year_ceiling,
            )
        day = self.selected_days[index]
        graph = self._dataset.load_day(day)
        if self.expected_sample_fingerprints is not None:
            try:
                expected = self.expected_sample_fingerprints[day]
            except KeyError as exc:
                raise RuntimeError(
                    f"temporal loader has no selected sample fingerprint for {day}"
                ) from exc
            # Verify in the worker before collate discards per-day boundaries.
            # No digest cache is used: every epoch must attest the GraphDay it
            # actually materialized, including labels, endpoints, and features.
            from cpv26.data.kbo_temporal_archive import _sample_fingerprint

            if _sample_fingerprint(graph) != expected:
                raise RuntimeError(
                    "materialized temporal sample differs from the selected CUDA plan: "
                    f"{day.isoformat()}"
                )
        return graph


def _graph_control_report(config: KBOTrainingConfig) -> dict[str, Any]:
    transform_algorithm_version = (
        f"identity_v{GRAPH_CONTROL_PROTOCOL_VERSION}"
        if config.graph_control == "intact"
        else f"day_local_source_destination_node_bijection_v{GRAPH_CONTROL_PROTOCOL_VERSION}"
    )
    payload: dict[str, Any] = {
        "mode": config.graph_control,
        "control_seed": config.graph_control_seed,
        "transform_algorithm_version": transform_algorithm_version,
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {**payload, "fingerprint": hashlib.sha256(encoded).hexdigest()}


def _validate_checkpoint_graph_control(
    state: Mapping[str, Any], config: KBOTrainingConfig
) -> dict[str, Any]:
    expected = _graph_control_report(config)
    saved = state.get("graph_control")
    if saved is None:
        if config.graph_control != "intact":
            raise ValueError("non-intact checkpoint is missing its graph-control protocol")
        return {**expected, "legacy_checkpoint_default": True}
    if saved != expected:
        raise ValueError(
            "checkpoint graph-control protocol differs from the training configuration"
        )
    return expected


def _prepare_graph_batch(
    batch: Mapping[str, Any], config: KBOTrainingConfig
) -> Mapping[str, Any]:
    """Apply one epoch-invariant graph control before pinning or device transfer."""

    if config.graph_control == "intact":
        return batch
    from cpv26.training.kbo_graph_diagnostic import (
        KBOGraphTransformSpec,
        transform_kbo_graph_batch,
    )

    transformed, _ = transform_kbo_graph_batch(
        batch,
        KBOGraphTransformSpec("permute_endpoints", seed=config.graph_control_seed),
    )
    return transformed


def _collate_loader_days(
    days: Sequence[GraphDay],
    *,
    config: KBOTrainingConfig,
    epoch: int,
    training: bool,
) -> Mapping[str, Any]:
    started = time.perf_counter()
    batch = collate_kbo_day_graphs(
        days,
        device="cpu",
        max_pa_per_day=config.max_pa_per_day if training else None,
        max_edges_per_route_per_day=config.max_edges_per_route_per_day,
        seed=config.seed + epoch if training else config.seed,
    )
    prepared = dict(_prepare_graph_batch(batch, config))
    prepared["_runtime_telemetry"] = {
        "collate_seconds": time.perf_counter() - started,
        "graph_days": len(days),
    }
    return prepared


def _temporal_variant(config: KBOTrainingConfig) -> str:
    if config.route_schedule == "full":
        return "full"
    if config.route_schedule == "node_only":
        return "node_only"
    raise ValueError("temporal_v7 production execution permits only full or node_only")


def _temporal_preflight_integer(
    value: Any, *, context: str, minimum: int = 0
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return int(value)


def _temporal_preflight_fraction(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite fraction")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{context} must be a finite fraction in (0, 1]")
    return result


def _validate_temporal_plan_rows(
    raw_plan: Mapping[str, Any],
    *,
    config: KBOTrainingConfig,
) -> list[Mapping[str, Any]]:
    raw_rows = raw_plan.get("ordered_batches")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("temporal CUDA preflight ordered batches are malformed")
    budgets = raw_plan.get("budgets")
    if not isinstance(budgets, Mapping) or set(budgets) != {"max_nodes", "max_edges"}:
        raise ValueError("temporal CUDA preflight batch budgets are malformed")
    max_nodes = _temporal_preflight_integer(
        budgets["max_nodes"], context="temporal max_nodes", minimum=1
    )
    max_edges = _temporal_preflight_integer(
        budgets["max_edges"], context="temporal max_edges", minimum=1
    )
    if raw_plan.get("batching_basis") != "node_and_edge_totals_only":
        raise ValueError("temporal CUDA preflight batching basis is unsupported")
    if raw_plan.get("fixed_day_count_cap") is not False:
        raise ValueError("temporal CUDA preflight must not impose a fixed day-count cap")
    if raw_plan.get("prefetch_depth") != 1:
        raise ValueError("temporal CUDA preflight must use exactly one-batch look-ahead")
    loader_runtime = raw_plan.get("loader_runtime")
    if not isinstance(loader_runtime, Mapping) or set(loader_runtime) != {
        "workers",
        "prefetch_factor",
        "persistent_workers",
        "loader_instances",
        "simultaneous_worker_pools",
        "total_worker_processes",
        "packed_transfers",
        "pin_memory",
    }:
        raise ValueError("temporal CUDA preflight loader runtime is malformed")
    workers = _temporal_preflight_integer(
        loader_runtime.get("workers"), context="temporal loader workers"
    )
    prefetch_factor = loader_runtime.get("prefetch_factor")
    if workers == 0:
        if prefetch_factor is not None or loader_runtime.get("persistent_workers") is not False:
            raise ValueError("zero-worker temporal loader has invalid prefetch settings")
    elif (
        isinstance(prefetch_factor, bool)
        or not isinstance(prefetch_factor, int)
        or prefetch_factor < 1
        or loader_runtime.get("persistent_workers") is not True
    ):
        raise ValueError("multiprocess temporal loader has invalid prefetch settings")
    if (
        loader_runtime.get("loader_instances") != 2
        or loader_runtime.get("simultaneous_worker_pools") != (2 if workers > 0 else 0)
        or loader_runtime.get("total_worker_processes") != workers * 2
    ):
        raise ValueError("temporal loader-pool residency is inconsistent")
    if (
        loader_runtime.get("packed_transfers") is not True
        or loader_runtime.get("pin_memory") is not False
    ):
        raise ValueError("temporal loader transport contract differs")

    rows: list[Mapping[str, Any]] = []
    seen_dates: set[str] = set()
    split_counts = {"train": 0, "validation": 0}
    seen_validation = False
    oversize_count = 0
    for index, value in enumerate(raw_rows):
        if not isinstance(value, Mapping):
            raise ValueError("temporal CUDA preflight batch row is malformed")
        row = value
        split = row.get("split")
        if split not in split_counts:
            raise ValueError("temporal CUDA preflight batch split is malformed")
        if split == "validation":
            seen_validation = True
        elif seen_validation:
            raise ValueError("temporal CUDA preflight train batch follows validation")
        split_index = _temporal_preflight_integer(
            row.get("split_batch_index"),
            context=f"temporal batch {index} split_batch_index",
        )
        if split_index != split_counts[split]:
            raise ValueError("temporal CUDA preflight split batch indices are not contiguous")
        split_counts[split] += 1
        dates = row.get("dates")
        fingerprints = row.get("sample_fingerprints")
        if (
            not isinstance(dates, list)
            or not dates
            or not isinstance(fingerprints, list)
            or len(fingerprints) != len(dates)
        ):
            raise ValueError("temporal CUDA preflight batch samples are malformed")
        for raw_day, fingerprint in zip(dates, fingerprints, strict=True):
            if not isinstance(raw_day, str):
                raise ValueError("temporal CUDA preflight batch date is malformed")
            try:
                parsed = date.fromisoformat(raw_day)
            except ValueError as exc:
                raise ValueError("temporal CUDA preflight batch date is malformed") from exc
            if (
                parsed.isoformat() != raw_day
                or raw_day in seen_dates
                or parsed.year == config.test_season
            ):
                raise ValueError(
                    "temporal CUDA preflight dates are duplicate, noncanonical, or held out"
                )
            seen_dates.add(raw_day)
            if (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
            ):
                raise ValueError("temporal CUDA preflight sample fingerprint is malformed")
        nodes = _temporal_preflight_integer(
            row.get("nodes"), context=f"temporal batch {index} nodes", minimum=1
        )
        edges = _temporal_preflight_integer(
            row.get("edges"), context=f"temporal batch {index} edges", minimum=1
        )
        oversize = row.get("oversize_single_day")
        if not isinstance(oversize, bool):
            raise ValueError("temporal CUDA preflight oversize marker is malformed")
        expected_oversize = len(dates) == 1 and (
            nodes > max_nodes or edges > max_edges
        )
        if oversize is not expected_oversize:
            raise ValueError("temporal CUDA preflight oversize marker is inconsistent")
        oversize_count += int(oversize)
        if not isinstance(row.get("prefetch_next"), bool):
            raise ValueError("temporal CUDA preflight prefetch marker is malformed")
        rows.append(row)

    for index, row in enumerate(rows):
        following = rows[index + 1] if index + 1 < len(rows) else None
        expected_prefetch = bool(
            following is not None
            and following["split"] == row["split"]
            and row["oversize_single_day"] is False
            and following["oversize_single_day"] is False
        )
        if row["prefetch_next"] is not expected_prefetch:
            raise ValueError("temporal CUDA preflight prefetch barrier is inconsistent")

    summaries = {
        "actual_batch_count": len(rows),
        "train_batch_count": split_counts["train"],
        "validation_batch_count": split_counts["validation"],
        "oversize_single_day_batches": oversize_count,
    }
    for name, expected in summaries.items():
        if _temporal_preflight_integer(
            raw_plan.get(name), context=f"temporal execution plan {name}"
        ) != expected:
            raise ValueError(f"temporal CUDA preflight {name} is inconsistent")
    physical = raw_plan.get("physical_batching")
    graph_days = [len(row["dates"]) for row in rows]
    effective: list[int] = []
    for split in ("train", "validation"):
        split_rows = [row for row in rows if row["split"] == split]
        for start in range(0, len(split_rows), config.accumulate_steps):
            effective.append(
                sum(
                    len(row["dates"])
                    for row in split_rows[start : start + config.accumulate_steps]
                )
            )
    expected_physical = {
        "unit": "graph_days",
        "physical_graph_days": numeric_distribution(graph_days),
        "effective_graph_days_per_optimizer_step": numeric_distribution(effective),
        "gradient_accumulation_steps": config.accumulate_steps,
        "data_parallel_workers": 1,
        "formula": (
            "effective graph-days = sum(dynamic physical graph-days in accumulation "
            "group) * data-parallel workers"
        ),
        "no_graph_or_event_dropped_by_batching": True,
    }
    if physical != expected_physical:
        raise ValueError("temporal physical/effective batch report is inconsistent")
    held_out = raw_plan.get("held_out_test")
    if not isinstance(held_out, Mapping) or dict(held_out) != {
        "season": config.test_season,
        "graph_days_loaded": False,
        "labels_loaded": False,
        "sealed": True,
    }:
        raise ValueError("temporal CUDA preflight does not prove held-out test sealing")
    return rows


def _validate_temporal_measurements(
    report: Mapping[str, Any],
    measured: Mapping[str, Any],
    raw_plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    runtime: Mapping[str, Any],
) -> None:
    if measured.get("execution_plan") != raw_plan:
        raise ValueError("temporal selected plan differs from its measured attempt")
    if measured.get("selected_for_training") is not True:
        raise ValueError("temporal selected attempt was not selected for training")
    if measured.get("all_actual_batches_measured") is not True:
        raise ValueError("temporal selected attempt did not measure every batch")
    if report.get("all_actual_batches_measured") is not True:
        raise ValueError("temporal adaptive report did not attest every selected batch")
    loader_autotune = measured.get("loader_autotune")
    if not isinstance(loader_autotune, Mapping) or loader_autotune.get("status") != "measured":
        raise ValueError("temporal selected attempt has no measured loader autotune")
    tuned_loader = loader_autotune.get("selected")
    planned_loader = raw_plan.get("loader_runtime")
    if not isinstance(tuned_loader, Mapping) or not isinstance(planned_loader, Mapping):
        raise ValueError("temporal loader autotune selection is malformed")
    for name in (
        "workers",
        "prefetch_factor",
        "persistent_workers",
        "loader_instances",
        "simultaneous_worker_pools",
        "total_worker_processes",
        "packed_transfers",
        "pin_memory",
    ):
        if tuned_loader.get(name) != planned_loader.get(name):
            raise ValueError("temporal loader autotune differs from the selected plan")
    if tuned_loader.get("host_memory_safe") is not True:
        raise ValueError("temporal loader autotune did not pass host-memory safety")
    host_memory = tuned_loader.get("host_memory_safety")
    if not isinstance(host_memory, Mapping) or host_memory.get("status") != "passed":
        raise ValueError("temporal loader host-memory evidence is missing or unsafe")
    resources = measured.get("resource_measurements")
    if not isinstance(resources, Mapping):
        raise ValueError("temporal selected attempt has no resource measurements")
    for name in (
        "host_inventory",
        "interval",
        "input_tensor_shapes_first_actual_batch",
        "stage_seconds",
        "throughput",
        "steady_cuda_allocated_bytes",
        "steady_cuda_reserved_bytes",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "physical_batching",
    ):
        if name not in resources:
            raise ValueError(f"temporal resource measurements are missing {name}")
    if resources.get("physical_batching") != raw_plan.get("physical_batching"):
        raise ValueError("temporal resource batch report differs from the plan")
    limit = _temporal_preflight_fraction(
        report.get("max_reserved_fraction"), context="temporal memory limit"
    )
    measured_limit = _temporal_preflight_fraction(
        measured.get("max_reserved_fraction"), context="temporal selected-attempt limit"
    )
    if not math.isclose(limit, measured_limit, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("temporal selected-attempt memory limit differs")
    if limit > 0.85:
        raise ValueError("temporal CUDA preflight memory limit exceeds 85%")
    measurements = measured.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != len(rows):
        raise ValueError("temporal CUDA preflight did not measure every selected batch")
    if _temporal_preflight_integer(
        measured.get("completed_actual_batch_count"),
        context="temporal completed batch count",
    ) != len(rows):
        raise ValueError("temporal CUDA preflight completed batch count is inconsistent")

    fractions: list[float] = []
    for index, (row, measurement) in enumerate(
        zip(rows, measurements, strict=True)
    ):
        if not isinstance(measurement, Mapping):
            raise ValueError("temporal CUDA preflight batch evidence is malformed")
        exact_fields = {
            "actual_batch_index": index,
            "split": row["split"],
            "split_batch_index": row["split_batch_index"],
            "dates": row["dates"],
            "nodes": row["nodes"],
            "edges": row["edges"],
            "oversize_single_day": row["oversize_single_day"],
            "prefetch_depth": raw_plan["prefetch_depth"],
        }
        if any(measurement.get(name) != expected for name, expected in exact_fields.items()):
            raise ValueError("temporal CUDA preflight measurement differs from its plan row")
        expected_next = None
        if row["prefetch_next"] is True:
            next_row = rows[index + 1]
            expected_next = {
                "split": next_row["split"],
                "dates": next_row["dates"],
                "nodes": next_row["nodes"],
                "edges": next_row["edges"],
                "oversize_single_day": next_row["oversize_single_day"],
            }
        if measurement.get("prefetched_next_batch") != expected_next:
            raise ValueError("temporal CUDA preflight prefetch evidence differs from its plan")
        allocated = _temporal_preflight_integer(
            measurement.get("peak_allocated_bytes"),
            context=f"temporal measurement {index} peak allocated bytes",
            minimum=1,
        )
        reserved = _temporal_preflight_integer(
            measurement.get("peak_reserved_bytes"),
            context=f"temporal measurement {index} peak reserved bytes",
            minimum=1,
        )
        total = _temporal_preflight_integer(
            measurement.get("total_memory_bytes"),
            context=f"temporal measurement {index} total memory bytes",
            minimum=1,
        )
        if allocated > reserved or reserved > total:
            raise ValueError("temporal CUDA preflight byte measurements are inconsistent")
        if total != runtime.get("total_memory_bytes"):
            raise ValueError("temporal CUDA preflight measured a different GPU memory size")
        fraction = _temporal_preflight_fraction(
            measurement.get("peak_reserved_fraction"),
            context=f"temporal measurement {index} peak reserved fraction",
        )
        if not math.isclose(fraction, reserved / total, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("temporal CUDA preflight byte/fraction evidence is inconsistent")
        if fraction > limit:
            raise ValueError("temporal CUDA preflight selected an over-limit batch")
        fractions.append(fraction)

    observed_peak = max(fractions)
    selected_peak = _temporal_preflight_fraction(
        measured.get("peak_reserved_fraction"), context="temporal selected-attempt peak"
    )
    report_peak = _temporal_preflight_fraction(
        report.get("peak_reserved_fraction"), context="temporal adaptive-report peak"
    )
    if not math.isclose(observed_peak, selected_peak, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("temporal selected-attempt peak differs from its measurements")
    if not math.isclose(observed_peak, report_peak, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("temporal adaptive-report peak differs from its measurements")


def _load_temporal_execution(
    dataset: KBOGraphDatasetLike,
    config: KBOTrainingConfig,
    report_path: str | Path,
    *,
    runtime: Mapping[str, Any],
) -> _TemporalExecution:
    """Load and bind one passed preflight artifact to this exact run."""

    path = Path(report_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"temporal CUDA preflight report does not exist: {path}")
    payload = path.read_bytes()
    try:
        report = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("temporal CUDA preflight report is not valid UTF-8 JSON") from exc
    if not isinstance(report, Mapping):
        raise ValueError("temporal CUDA preflight report must contain an object")
    variant = _temporal_variant(config)
    from cpv26.training.kbo_temporal_preflight import (
        TEMPORAL_PREFLIGHT_PROTOCOL,
        TEMPORAL_PREFLIGHT_PROTOCOL_VERSION,
        validate_temporal_execution_plan,
    )

    if report.get("protocol") != f"{TEMPORAL_PREFLIGHT_PROTOCOL}_adaptive":
        raise ValueError("temporal training requires the adaptive CUDA preflight protocol")
    if report.get("protocol_version") != TEMPORAL_PREFLIGHT_PROTOCOL_VERSION:
        raise ValueError("temporal CUDA preflight protocol version differs")
    rows = validate_temporal_execution_plan(
        report,
        dataset_fingerprint=str(dataset.manifest["fingerprint"]),
        variant=variant,
    )
    raw_plan = report["execution_plan"]
    assert isinstance(raw_plan, Mapping)
    raw_rows = _validate_temporal_plan_rows(raw_plan, config=config)
    if len(raw_rows) != len(rows):
        raise ValueError("temporal CUDA preflight ordered batches are malformed")

    selected_attempt = report.get("selected_attempt")
    attempts = report.get("attempts")
    if (
        isinstance(selected_attempt, bool)
        or not isinstance(selected_attempt, int)
        or not isinstance(attempts, list)
        or not 1 <= selected_attempt <= len(attempts)
    ):
        raise ValueError("temporal CUDA preflight selected attempt is malformed")
    measured = attempts[selected_attempt - 1]
    if not isinstance(measured, Mapping) or measured.get("status") != "passed":
        raise ValueError("temporal CUDA preflight selected attempt did not pass")
    if measured.get("protocol") != TEMPORAL_PREFLIGHT_PROTOCOL:
        raise ValueError("temporal selected attempt has the wrong protocol")
    if measured.get("protocol_version") != TEMPORAL_PREFLIGHT_PROTOCOL_VERSION:
        raise ValueError("temporal selected attempt protocol version differs")
    raw_saved_config = measured.get("configuration")
    if not isinstance(raw_saved_config, Mapping):
        raise ValueError("temporal CUDA preflight is missing its training configuration")
    saved_config = asdict(KBOTrainingConfig.from_dict(raw_saved_config))
    expected_config = asdict(config)
    if variant == "node_only":
        saved_config["route_schedule"] = "node_only"
    if saved_config != expected_config:
        raise ValueError("temporal CUDA preflight configuration differs from training")
    saved_runtime = measured.get("runtime")
    if not isinstance(saved_runtime, Mapping):
        raise ValueError("temporal CUDA preflight is missing its runtime signature")
    for name in (
        "device",
        "gpu_name",
        "total_memory_bytes",
        "compute_capability",
        "torch_version",
        "cuda_runtime",
        "precision",
    ):
        if saved_runtime.get(name) != runtime.get(name):
            raise ValueError(f"temporal CUDA preflight runtime.{name} differs from training")

    planned_dates = [day for batch in rows for day in batch]
    splits = _split_days(dataset, config)
    expected_dates = [*splits["train"], *splits["validation"]]
    if planned_dates != expected_dates or any(
        day.year == config.test_season for day in planned_dates
    ):
        raise ValueError("temporal CUDA preflight plan does not exactly cover train/validation")
    _validate_temporal_measurements(
        report,
        measured,
        raw_plan,
        raw_rows,
        runtime=runtime,
    )
    return _TemporalExecution(
        report_path=path,
        report_sha256=hashlib.sha256(payload).hexdigest(),
        plan_fingerprint=str(raw_plan["plan_fingerprint"]),
        variant=variant,
        rows=tuple(dict(row) for row in raw_rows),
        loader_workers=int(raw_plan["loader_runtime"]["workers"]),
        loader_prefetch_factor=raw_plan["loader_runtime"]["prefetch_factor"],
        loader_persistent_workers=bool(
            raw_plan["loader_runtime"]["persistent_workers"]
        ),
        preflight_resources=dict(measured["resource_measurements"]),
        loader_autotune=dict(measured["loader_autotune"]),
    )


def _validate_temporal_batch(
    batch: Mapping[str, Any], row: Mapping[str, Any]
) -> Mapping[str, Any]:
    dates = list(batch.get("day_ids", ()))
    if dates != list(row["dates"]):
        raise RuntimeError("temporal loader dates differ from the selected CUDA plan")
    nodes = sum(int(values.shape[0]) for values in batch["node_features"].values())
    edges = sum(int(route.num_edges) for route in batch["routes"])
    if (nodes, edges) != (int(row["nodes"]), int(row["edges"])):
        raise RuntimeError("temporal loader topology differs from the selected CUDA plan")
    return batch


def _planned_device_batches(
    batches: Any,
    rows: Sequence[Mapping[str, Any]],
    device: Any,
    *,
    observer: Callable[[str, Any], None] | None = None,
) -> Any:
    """Consume one DataLoader using the preflight's exact prefetch barriers."""

    iterator = iter(batches)
    start = 0
    while start < len(rows):
        end = start
        while end + 1 < len(rows) and rows[end].get("prefetch_next") is True:
            end += 1
        segment_rows = rows[start : end + 1]

        def checked_segment(
            selected_rows: Sequence[Mapping[str, Any]] = segment_rows,
        ) -> Any:
            for row in selected_rows:
                try:
                    raw = next(iterator)
                except StopIteration as exc:
                    raise RuntimeError("temporal loader ended before its selected plan") from exc
                yield _validate_temporal_batch(raw, row)

        yield from prefetch_batches(
            checked_segment(), device, mover=_move, observer=observer
        )
        start = end + 1
        if start < len(rows):
            # A false prefetch marker is a GPU-storage boundary, not just a
            # fresh Python generator.  The previous segment's last compute is
            # asynchronous, so wait before the next copy stream can allocate
            # the following (often oversize) segment.
            torch, _ = require_torch()
            torch.cuda.current_stream(device).synchronize()
    try:
        next(iterator)
    except StopIteration:
        return
    raise RuntimeError("temporal loader produced more batches than its selected plan")


def _loader(
    directory: Path,
    days: Sequence[date],
    config: KBOTrainingConfig,
    *,
    epoch: int,
    training: bool,
    packed_transfers: bool = True,
    planned_rows: Sequence[Mapping[str, Any]] | None = None,
    workers_override: int | None = None,
    prefetch_factor_override: int | None = None,
    persistent_workers_override: bool | None = None,
) -> Any:
    torch, _ = require_torch()
    selected_workers = config.workers if workers_override is None else workers_override
    if (
        isinstance(selected_workers, bool)
        or not isinstance(selected_workers, int)
        or selected_workers < 0
    ):
        raise ValueError("loader workers must be a non-negative integer")
    selected_prefetch = (
        2 if prefetch_factor_override is None else prefetch_factor_override
    )
    if selected_workers > 0 and (
        isinstance(selected_prefetch, bool)
        or not isinstance(selected_prefetch, int)
        or selected_prefetch < 1
    ):
        raise ValueError("loader prefetch_factor must be positive when workers are used")
    persistent = (
        selected_workers > 0
        if persistent_workers_override is None
        else persistent_workers_override
    )
    if not isinstance(persistent, bool) or (persistent and selected_workers == 0):
        raise ValueError("persistent workers require at least one loader worker")
    generator = torch.Generator().manual_seed(config.seed + epoch)
    ordered_days = sorted(days) if config.chronological or not training else days
    common: dict[str, Any] = {
        "num_workers": selected_workers,
        # Workers start after model/CUDA initialization. Do not fork its
        # threaded runtime; keep this choice local to this DataLoader.
        "multiprocessing_context": "spawn" if selected_workers > 0 else None,
        # Packed transfer creates one private pinned buffer per dtype.  Pinning
        # every collated leaf here as well would copy the same batch into pinned
        # memory twice before H2D.  The unpacked compatibility path still needs
        # DataLoader pinning for genuinely asynchronous leaf transfers.
        "pin_memory": config.device.startswith("cuda") and not packed_transfers,
        "collate_fn": partial(
            _collate_loader_days,
            config=config,
            epoch=epoch,
            training=training,
        ),
    }
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        # Some unit/profile callers construct only the loader shell.  A real
        # dataset is still validated when _DayDataset first loads an item.
        manifest = {}
    if isinstance(manifest, dict) and manifest.get("graph_schema") == "temporal_v7":
        if planned_rows is not None:
            planned_dates = [
                date.fromisoformat(str(value))
                for row in planned_rows
                for value in row["dates"]
            ]
            if planned_dates != list(ordered_days):
                raise ValueError("selected temporal batches do not exactly cover loader days")
            index_by_day = {day: index for index, day in enumerate(ordered_days)}
            fingerprint_by_day = {
                date.fromisoformat(str(day_id)): str(fingerprint)
                for row in planned_rows
                for day_id, fingerprint in zip(
                    row["dates"], row["sample_fingerprints"], strict=True
                )
            }
            if set(fingerprint_by_day) != set(ordered_days):
                raise ValueError(
                    "selected temporal sample fingerprints do not exactly cover loader days"
                )
            selected_batch_sampler = tuple(
                tuple(index_by_day[date.fromisoformat(str(value))] for value in row["dates"])
                for row in planned_rows
            )
            if selected_workers > 0:
                common.update(
                    persistent_workers=persistent,
                    prefetch_factor=selected_prefetch,
                )
            return torch.utils.data.DataLoader(
                _DayDataset(
                    directory,
                    ordered_days,
                    expected_sample_fingerprints=fingerprint_by_day,
                    label_year_ceiling=config.validation_season,
                ),
                batch_sampler=selected_batch_sampler,
                **common,
            )
        from cpv26.training.kbo_temporal_batching import (
            TemporalBudgetBatchSampler,
            load_temporal_sample_sizes,
        )

        batching = manifest.get("temporal_batching")
        if not isinstance(batching, Mapping):
            raise ValueError("temporal_v7 manifest is missing its batching contract")
        sizes = load_temporal_sample_sizes(
            directory,
            dataset_fingerprint=str(manifest["fingerprint"]),
            sampling_policy_fingerprint=str(manifest["sampling_policy_fingerprint"]),
        )
        budget_batch_sampler = TemporalBudgetBatchSampler(
            ordered_days,
            sizes,
            max_nodes=int(batching["max_nodes_per_batch"]),
            max_edges=int(batching["max_edges_per_batch"]),
            max_days=min(config.batch_days, int(batching["max_days_per_batch"])),
            shuffle=training and not config.chronological,
            seed=config.seed + epoch,
        )
        if selected_workers > 0:
            common.update(
                persistent_workers=persistent,
                prefetch_factor=selected_prefetch,
            )
        return torch.utils.data.DataLoader(
            _DayDataset(
                directory,
                ordered_days,
                label_year_ceiling=config.validation_season,
            ),
            batch_sampler=budget_batch_sampler,
            **common,
        )
    return torch.utils.data.DataLoader(
        _DayDataset(
            directory,
            ordered_days,
            label_year_ceiling=config.validation_season,
        ),
        batch_size=config.batch_days,
        shuffle=training and not config.chronological,
        generator=generator,
        **common,
    )


def _split_days(
    dataset: KBOGraphDatasetLike,
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
    dataset: KBOGraphDatasetLike, splits: Mapping[str, Sequence[date]]
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


def _sealed_split_summary(days: Sequence[date]) -> dict[str, Any]:
    """Describe a held-out split without reading any graph or label payload."""

    return {
        "seasons": sorted({day.year for day in days}),
        "days": len(days),
        "date_start": min(days).isoformat() if days else None,
        "date_end": max(days).isoformat() if days else None,
        "labels_or_graphs_loaded": False,
    }


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


def _training_counts(
    batch: Mapping[str, Any], *, include_boxscore: bool = False
) -> dict[str, Any]:
    """Keep count reductions on-device until the periodic/epoch sync boundary."""

    counts: dict[str, Any] = {
        "match": batch["match_targets"].numel(),
        "live_hit": batch["live_hit_pa"].numel(),
        "pa": batch["pa_targets"].numel(),
        "run": batch["match_targets"].numel(),
    }
    if include_boxscore:
        counts["box_pa"] = batch["box_pa_counts"].long().sum()
        counts["box_pitch"] = batch["box_pitch_mask"].long().sum()
    return counts


def _resolved_route_schedule(
    config: KBOTrainingConfig,
) -> tuple[tuple[str, ...], ...] | None:
    if config.route_schedule == "full":
        return None
    if config.route_schedule == "staged":
        return (_STAGED_FIRST_ROUTE_GATE_KEYS,) + (_CORE_ROUTE_GATE_KEYS,) * (
            config.layers - 1
        )
    if config.route_schedule == "core":
        return (_CORE_ROUTE_GATE_KEYS,) * config.layers
    return ((),) * config.layers


def _model_config(dataset: KBOGraphDatasetLike, config: KBOTrainingConfig) -> KBORelGNNConfig:
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
        route_message_normalization=config.route_message_normalization,
        route_schedule=_resolved_route_schedule(config),
        route_edge_chunk_size=config.route_edge_chunk_size,
        activation_checkpointing=config.activation_checkpointing,
        compact_kbo_channels=config.compact_kbo_channels,
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


def _parameter_contract(model: Any, optimizer: Any) -> dict[str, Any]:
    """Prove that AdamW covers every trainable tensor exactly once."""

    named = dict(model.named_parameters())
    trainable = {name: parameter for name, parameter in named.items() if parameter.requires_grad}
    optimizer_groups = optimizer_parameter_names(model, optimizer)
    optimizer_names = [name for group in optimizer_groups for name in group]
    missing = sorted(set(trainable).difference(optimizer_names))
    duplicate_count = len(optimizer_names) - len(set(optimizer_names))
    if missing or duplicate_count:
        raise ValueError(
            "optimizer does not cover every trainable parameter exactly once: "
            f"missing={missing}, duplicates={duplicate_count}"
        )
    return {
        "parameter_count": sum(parameter.numel() for parameter in named.values()),
        "parameter_tensors": len(named),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable.values()
        ),
        "trainable_parameter_tensors": len(trainable),
        "optimizer_parameter_count": sum(
            named[name].numel() for name in optimizer_names if named[name].requires_grad
        ),
        "optimizer_parameter_tensors": sum(
            int(named[name].requires_grad) for name in optimizer_names
        ),
        "optimizer_covers_all_trainable": True,
        "optimizer_missing_trainable_parameters": [],
        "architecture": model.architecture_contract(),
    }


def _record_gradient_connections(model: Any, connected: set[str]) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            connected.add(name)


def _gradient_connectivity_report(model: Any, connected: set[str]) -> dict[str, Any]:
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing = sorted(set(trainable).difference(connected))
    return {
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable.values()
        ),
        "gradient_connected_parameter_tensors": len(connected),
        "gradient_connected_parameter_count": sum(
            trainable[name].numel() for name in connected
        ),
        "all_trainable_parameters_received_gradient": not missing,
        "parameters_without_gradient": missing,
    }


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
        "route_message_normalization": config.route_message_normalization,
        "route_schedule_preset": config.route_schedule,
        "route_edge_chunk_size": config.route_edge_chunk_size,
        "route_edge_chunking_is_lossless": True,
        "resolved_route_schedule": (
            [list(layer) for layer in model_config.route_schedule]
            if model_config.route_schedule is not None
            else None
        ),
        "graph_control": _graph_control_report(config),
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
            "activation_checkpointing": config.activation_checkpointing,
            "compact_kbo_channels": config.compact_kbo_channels,
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


def _cuda_stage_start(torch: Any, device: Any) -> Any | None:
    if device.type != "cuda":
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _cuda_stage_finish(
    torch: Any,
    device: Any,
    started: Any | None,
    destination: list[tuple[Any, Any]],
) -> None:
    if device.type != "cuda" or started is None:
        return
    finished = torch.cuda.Event(enable_timing=True)
    finished.record()
    destination.append((started, finished))


def _cuda_event_seconds(events: Sequence[tuple[Any, Any]]) -> float | None:
    if not events:
        return None
    return sum(float(start.elapsed_time(end)) for start, end in events) / 1000.0


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
    planned_rows: Sequence[Mapping[str, Any]] | None = None,
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
    def prepared_batches() -> Any:
        for raw_batch in loader:
            prepared_batch = (
                batch_transform(raw_batch) if batch_transform is not None else raw_batch
            )
            if diagnostics is not None:
                begin_batch = getattr(diagnostics, "begin_batch", None)
                if callable(begin_batch):
                    begin_batch(prepared_batch)
            yield prepared_batch

    # Diagnostic observers consume each CPU topology before its corresponding
    # model call, so retain their sequential path.  Normal training/evaluation
    # uses one-batch CUDA look-ahead on a dedicated copy stream.
    device_batches: Any
    if diagnostics is not None or device.type != "cuda":
        if planned_rows is not None and diagnostics is not None:
            raise ValueError("temporal selected plans do not support diagnostic transforms")
        device_batches = map(partial(_move, device=device), prepared_batches())
    elif planned_rows is not None:
        device_batches = _planned_device_batches(prepared_batches(), planned_rows, device)
    else:
        device_batches = prefetch_batches(prepared_batches(), device, mover=_move)
    with torch.inference_mode():
        for batch in device_batches:
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
                    for index, query_id in enumerate(batch["box_pa_query_ids"]):
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
                    for index, query_id in enumerate(batch["box_pitch_query_ids"]):
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
                    query_ids = batch[f"{name}_query_ids"]
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
            # The prefetcher deliberately overlaps this current device batch
            # with at most one next device batch.  Release all other iteration
            # locals promptly so the two-slot pipeline is the only overlap.
            if include_boxscore:
                del box_probabilities
            del labels, values, losses, outputs, batch
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
    dataset: KBOGraphDatasetLike,
    dataset_directory: Path,
    config: KBOTrainingConfig,
    model_config: KBORelGNNConfig,
    device: Any,
    epoch: int,
    global_step: int,
    best_score: float,
    best_epoch: int,
    stale_epochs: int,
    skipped_optimizer_steps: int,
    history: Sequence[Mapping[str, Any]],
    initial_model_state_sha256: str,
    parameter_count: int,
    trainable_parameter_count: int,
    parameter_contract: Mapping[str, Any],
    temporal_execution: _TemporalExecution | None,
) -> dict[str, Any]:
    torch, _ = require_torch()
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "dataset_fingerprint": dataset.manifest["fingerprint"],
        "dataset_directory": str(dataset_directory),
        "model_config": model_config.to_dict(),
        "training_config": asdict(config),
        "graph_control": _graph_control_report(config),
        "initial_model_state_sha256": initial_model_state_sha256,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "parameter_contract": dict(parameter_contract),
        "temporal_execution": (
            temporal_execution.lineage() if temporal_execution is not None else None
        ),
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
        "selected_cuda_device": str(device) if device.type == "cuda" else None,
        "selected_cuda_rng_state": (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        ),
        # Retain the legacy field so older readers can still inspect new checkpoints.
        # New resumes use the selected-device state above and do not depend on
        # the number of devices visible to the current process.
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


def _restore_cuda_rng_state(torch: Any, state: Mapping[str, Any], device: Any) -> None:
    """Restore the saved training device RNG without requiring the old device topology."""

    if device.type != "cuda":
        return
    if "selected_cuda_rng_state" in state:
        selected_state = state["selected_cuda_rng_state"]
        if selected_state is None:
            raise ValueError("CUDA resume checkpoint has no selected-device RNG state")
    else:
        legacy_states = state.get("cuda_rng_states")
        if not isinstance(legacy_states, (list, tuple)) or not legacy_states:
            raise ValueError("legacy CUDA checkpoint has no restorable CUDA RNG state")
        previous_device = torch.device(
            KBOTrainingConfig.from_dict(state["training_config"]).device
        )
        previous_index = previous_device.index
        if (
            previous_device.type != "cuda"
            or previous_index is None
            or previous_index < 0
            or previous_index >= len(legacy_states)
        ):
            raise ValueError(
                "legacy CUDA checkpoint has no RNG state for its configured device"
            )
        selected_state = legacy_states[previous_index]
    torch.cuda.set_rng_state(selected_state, device=device)


def train_kbo_relgnn(
    dataset_directory: str | Path,
    run_directory: str | Path,
    *,
    config: KBOTrainingConfig | None = None,
    resume: str | Path | None = None,
    temporal_preflight_report: str | Path | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train on configured past seasons and select with a later validation season."""
    torch, _ = require_torch()
    options = config or KBOTrainingConfig()
    device, dtype, runtime = _device_and_precision(options.device, options.amp)
    directory = Path(dataset_directory).expanduser().resolve()
    output = Path(run_directory).expanduser().resolve()
    dataset = open_kbo_graph_dataset(
        directory,
        label_year_ceiling=options.validation_season,
    )
    resource_inventory = host_resource_inventory(
        torch, device, dataset_directory=directory
    )
    temporal_schema = dataset.manifest.get("graph_schema") == "temporal_v7"
    if temporal_schema != (temporal_preflight_report is not None):
        raise ValueError(
            "temporal_v7 training requires exactly one passed adaptive CUDA preflight report"
        )
    temporal_execution = (
        _load_temporal_execution(
            dataset,
            options,
            temporal_preflight_report,
            runtime=runtime,
        )
        if temporal_preflight_report is not None
        else None
    )
    splits = _split_days(dataset, options)
    split_summary = _split_summary(
        dataset,
        {name: splits[name] for name in ("train", "validation")},
    )
    split_summary["test"] = _sealed_split_summary(splits["test"])
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
    initial_model_state_sha256 = _model_state_sha256(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.to(device)
    optimizer = make_adamw(
        model,
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    parameter_contract = _parameter_contract(model, optimizer)
    trainable_parameter_count = int(parameter_contract["trainable_parameter_count"])
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
        _validate_checkpoint_graph_control(state, options)
        _resume_compatible(state, options)
        saved_initial_hash = state.get("initial_model_state_sha256")
        if saved_initial_hash is not None and saved_initial_hash != initial_model_state_sha256:
            raise ValueError("checkpoint initial model state differs from the configured seed")
        saved_parameter_count = state.get("parameter_count")
        if saved_parameter_count is not None and int(saved_parameter_count) != parameter_count:
            raise ValueError("checkpoint parameter count differs from the current model")
        if state.get("parameter_contract") != parameter_contract:
            raise ValueError(
                "checkpoint predates or differs from the verified trainable-parameter contract"
            )
        if int(state.get("trainable_parameter_count", -1)) != trainable_parameter_count:
            raise ValueError("checkpoint trainable parameter count differs from the current model")
        expected_temporal = (
            temporal_execution.lineage() if temporal_execution is not None else None
        )
        if state.get("temporal_execution") != expected_temporal:
            raise ValueError("checkpoint temporal execution plan differs from the CUDA gate")
        model.load_state_dict(state["model"])
        optimizer = make_adamw(
            model, learning_rate=options.learning_rate, weight_decay=options.weight_decay,
            checkpoint=state,
        )
        if scaler.is_enabled() and state["scaler"]:
            scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng_state"])
        random.setstate(state["python_rng_state"])
        _restore_cuda_rng_state(torch, state, device)
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
                "parameter_contract": parameter_contract,
                "training_policies": _policy_report(options, model_config),
                "runtime": runtime,
                "resource_inventory": resource_inventory,
                "temporal_execution": (
                    temporal_execution.lineage()
                    if temporal_execution is not None
                    else None
                ),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    progress(
        f"RelGNN device={device} precision={runtime['precision']} "
        f"parameters={parameter_count:,}; trainable={trainable_parameter_count:,}; "
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
    train_plan_rows = (
        temporal_execution.split_rows("train") if temporal_execution is not None else None
    )
    validation_plan_rows = (
        temporal_execution.split_rows("validation")
        if temporal_execution is not None
        else None
    )
    if temporal_execution is not None:
        reusable_train_loader = _loader(
            directory,
            splits["train"],
            options,
            epoch=0,
            training=True,
            planned_rows=train_plan_rows,
            workers_override=temporal_execution.loader_workers,
            prefetch_factor_override=temporal_execution.loader_prefetch_factor,
            persistent_workers_override=(
                temporal_execution.loader_persistent_workers
            ),
        )
        reusable_validation_loader = _loader(
            directory,
            splits["validation"],
            options,
            epoch=0,
            training=False,
            planned_rows=validation_plan_rows,
            workers_override=temporal_execution.loader_workers,
            prefetch_factor_override=temporal_execution.loader_prefetch_factor,
            persistent_workers_override=(
                temporal_execution.loader_persistent_workers
            ),
        )
    else:
        reusable_train_loader = None
        reusable_validation_loader = None
    for epoch in range(start_epoch, options.epochs):
        started = time.monotonic()
        model.train()
        loader = reusable_train_loader or _loader(
            directory, splits["train"], options, epoch=epoch, training=True
        )
        optimizer.zero_grad(set_to_none=True)
        task_names = ("match", "live_hit", "pa", "run") + (
            ("box_pa", "box_pitch") if model_config.include_boxscore_heads else ()
        )
        sums = dict.fromkeys(task_names, 0.0)
        counts = dict.fromkeys(task_names, 0)
        loss_sum_tensors = {
            name: torch.zeros((), dtype=torch.float64, device=device)
            for name in task_names
        }
        count_tensors = {
            name: torch.zeros((), dtype=torch.int64, device=device)
            for name in task_names
        }
        gradient_groups = _gradient_parameter_groups(model)
        gradient_steps = dict.fromkeys(gradient_groups, 0)
        gradient_clipped_tensors = {
            name: torch.zeros((), dtype=torch.int64, device=device)
            for name in gradient_groups
        }
        gradient_nonfinite_tensors = {
            name: torch.zeros((), dtype=torch.int64, device=device)
            for name in gradient_groups
        }
        gradient_max_tensors = {
            name: torch.zeros((), dtype=torch.float64, device=device)
            for name in gradient_groups
        }
        all_losses_finite = torch.ones((), dtype=torch.bool, device=device)
        all_gradients_finite = torch.ones((), dtype=torch.bool, device=device)
        gradient_connected_parameters: set[str] = set()
        epoch_resource_start = resource_snapshot(torch, device)
        stage_host_seconds = {
            "source_wait_seconds": 0.0,
            "h2d_host_dispatch_seconds": 0.0,
            "forward_and_loss_host_seconds": 0.0,
            "backward_host_seconds": 0.0,
            "optimizer_host_seconds": 0.0,
            "collate_worker_seconds": 0.0,
            "periodic_synchronization_host_seconds": 0.0,
            "epoch_final_synchronization_host_seconds": 0.0,
        }
        cuda_stage_events: dict[str, list[tuple[Any, Any]]] = {
            "h2d": [],
            "forward_and_loss": [],
            "backward": [],
            "optimizer": [],
        }
        first_input_shapes: dict[str, Any] | None = None
        physical_graph_days: list[int] = []
        effective_graph_days: list[int] = []
        current_accumulated_graph_days = 0
        processed_nodes = 0
        processed_edges = 0
        steady_allocated: list[int] = []
        steady_reserved: list[int] = []
        utilization_samples: list[Mapping[str, Any]] = []
        post_fetch_step_host_seconds: list[float] = []

        def observe_transfer(
            name: str,
            value: Any,
            host_seconds: dict[str, float] = stage_host_seconds,
            event_groups: dict[str, list[tuple[Any, Any]]] = cuda_stage_events,
        ) -> None:
            if name in host_seconds:
                host_seconds[name] += float(value)
            elif name == "h2d_cuda_event":
                start_event, end_event = value
                event_groups["h2d"].append((start_event, end_event))

        if device.type == "cuda" and train_plan_rows is not None:
            source_batches = _planned_device_batches(
                loader,
                train_plan_rows,
                device,
                observer=observe_transfer,
            )
        elif device.type == "cuda":
            source_batches = prefetch_batches(
                loader,
                device,
                mover=_move,
                observer=observe_transfer,
            )
        else:
            source_batches = loader
        for index, raw_or_device_batch in enumerate(source_batches):
            post_fetch_step_started = time.perf_counter()
            batch = (
                raw_or_device_batch
                if device.type == "cuda"
                else _move(raw_or_device_batch, device)
            )
            if first_input_shapes is None:
                first_input_shapes = tensor_shape_manifest(batch, torch)
            batch_telemetry = batch.get("_runtime_telemetry", {})
            stage_host_seconds["collate_worker_seconds"] += float(
                batch_telemetry.get("collate_seconds", 0.0)
            )
            graph_days = len(batch.get("day_ids", ()))
            physical_graph_days.append(graph_days)
            current_accumulated_graph_days += graph_days
            processed_nodes += sum(
                int(values.shape[0]) for values in batch["node_features"].values()
            )
            processed_edges += sum(int(route.num_edges) for route in batch["routes"])
            forward_host_started = time.perf_counter()
            forward_cuda_started = _cuda_stage_start(torch, device)
            with torch.autocast(device.type, enabled=dtype is not None, dtype=dtype):
                losses = _losses(model(batch), batch, options)
            _cuda_stage_finish(
                torch,
                device,
                forward_cuda_started,
                cuda_stage_events["forward_and_loss"],
            )
            stage_host_seconds["forward_and_loss_host_seconds"] += (
                time.perf_counter() - forward_host_started
            )
            with torch.no_grad():
                all_losses_finite.logical_and_(torch.isfinite(losses["loss"]))
            group_start = (index // options.accumulate_steps) * options.accumulate_steps
            group_size = min(options.accumulate_steps, len(loader) - group_start)
            backward_host_started = time.perf_counter()
            backward_cuda_started = _cuda_stage_start(torch, device)
            scaler.scale(losses["loss"] / group_size).backward()
            _cuda_stage_finish(
                torch,
                device,
                backward_cuda_started,
                cuda_stage_events["backward"],
            )
            stage_host_seconds["backward_host_seconds"] += (
                time.perf_counter() - backward_host_started
            )
            _record_gradient_connections(model, gradient_connected_parameters)
            batch_counts = _training_counts(
                batch, include_boxscore=model_config.include_boxscore_heads
            )
            with torch.no_grad():
                for name, count in batch_counts.items():
                    loss_sum_tensors[name].add_(
                        losses[f"{name}_loss"].detach().to(torch.float64) * count
                    )
                    count_tensors[name].add_(count)
            if (index + 1) % options.accumulate_steps == 0 or index + 1 == len(loader):
                optimizer_host_started = time.perf_counter()
                optimizer_cuda_started = _cuda_stage_start(torch, device)
                scaler.unscale_(optimizer)
                norms = _clip_gradient_norms(model, options.gradient_clip)
                with torch.no_grad():
                    for name, norm in norms.items():
                        gradient_steps[name] += 1
                        value = norm.detach().to(torch.float64)
                        finite = torch.isfinite(value)
                        gradient_clipped_tensors[name].add_(
                            torch.logical_and(finite, value > options.gradient_clip)
                        )
                        gradient_nonfinite_tensors[name].add_(torch.logical_not(finite))
                        gradient_max_tensors[name].copy_(
                            torch.maximum(
                                gradient_max_tensors[name],
                                torch.where(finite, value, torch.zeros_like(value)),
                            )
                        )
                        all_gradients_finite.logical_and_(finite)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() < previous_scale:
                    skipped_optimizer_steps += 1
                else:
                    global_step += 1
                _cuda_stage_finish(
                    torch,
                    device,
                    optimizer_cuda_started,
                    cuda_stage_events["optimizer"],
                )
                stage_host_seconds["optimizer_host_seconds"] += (
                    time.perf_counter() - optimizer_host_started
                )
                effective_graph_days.append(current_accumulated_graph_days)
                current_accumulated_graph_days = 0
            if device.type == "cuda":
                steady_allocated.append(int(torch.cuda.memory_allocated(device)))
                steady_reserved.append(int(torch.cuda.memory_reserved(device)))
            if (index + 1) % 10 == 0 or index + 1 == len(loader):
                synchronization_started = time.perf_counter()
                safety_and_loss = (
                    torch.stack(
                        (
                            all_losses_finite.to(torch.float32),
                            all_gradients_finite.to(torch.float32),
                            losses["loss"].detach().to(torch.float32),
                        )
                    )
                    .cpu()
                    .tolist()
                )
                stage_host_seconds["periodic_synchronization_host_seconds"] += (
                    time.perf_counter() - synchronization_started
                )
                if not bool(safety_and_loss[0]):
                    raise FloatingPointError("non-finite RelGNN training loss")
                if not scaler.is_enabled() and not bool(safety_and_loss[1]):
                    raise FloatingPointError("non-finite RelGNN gradients")
                utilization_samples.append(resource_snapshot(torch, device))
                progress(
                    f"epoch {epoch + 1}/{options.epochs} batch {index + 1}/{len(loader)} "
                    f"loss={float(safety_and_loss[2]):.4f}"
                )
            post_fetch_step_host_seconds.append(
                time.perf_counter() - post_fetch_step_started
            )
            # Retain only the prefetcher's intentional current/next batch pair.
            # This also releases the last training batch before validation.
            del losses, batch, raw_or_device_batch
        if current_accumulated_graph_days:
            raise RuntimeError("effective batch accounting did not close at optimizer step")
        if device.type == "cuda":
            synchronization_started = time.perf_counter()
            torch.cuda.synchronize(device)
            stage_host_seconds["epoch_final_synchronization_host_seconds"] += (
                time.perf_counter() - synchronization_started
            )
        aggregate_values = (
            torch.stack(
                tuple(loss_sum_tensors[name] for name in task_names)
                + tuple(count_tensors[name].to(torch.float64) for name in task_names)
                + tuple(
                    gradient_clipped_tensors[name].to(torch.float64)
                    for name in gradient_groups
                )
                + tuple(
                    gradient_nonfinite_tensors[name].to(torch.float64)
                    for name in gradient_groups
                )
                + tuple(gradient_max_tensors[name] for name in gradient_groups)
            )
            .cpu()
            .tolist()
        )
        cursor = 0
        for name in task_names:
            sums[name] = float(aggregate_values[cursor])
            cursor += 1
        for name in task_names:
            counts[name] = int(aggregate_values[cursor])
            cursor += 1
        clipped_counts: dict[str, int] = {}
        nonfinite_counts: dict[str, int] = {}
        maximum_norms: dict[str, float] = {}
        for name in gradient_groups:
            clipped_counts[name] = int(aggregate_values[cursor])
            cursor += 1
        for name in gradient_groups:
            nonfinite_counts[name] = int(aggregate_values[cursor])
            cursor += 1
        for name in gradient_groups:
            maximum_norms[name] = float(aggregate_values[cursor])
            cursor += 1
        gradient_audit = {
            name: {
                "steps": int(gradient_steps[name]),
                "clipped_steps": clipped_counts[name],
                "nonfinite_steps": nonfinite_counts[name],
                "max_finite_preclip_norm": maximum_norms[name],
            }
            for name in gradient_groups
        }
        epoch_resource_end = resource_snapshot(torch, device)
        interval_resources = summarize_resource_interval(
            epoch_resource_start,
            epoch_resource_end,
            allowed_cpu_count=int(resource_inventory["allowed_cpu_count"]),
        )
        gpu_utilization_values: list[float] = []
        gpu_memory_utilization_values: list[float] = []
        for utilization_sample in utilization_samples:
            raw_gpu_utilization = utilization_sample.get("gpu_utilization_percent")
            if isinstance(raw_gpu_utilization, (int, float)):
                gpu_utilization_values.append(float(raw_gpu_utilization))
            raw_gpu_memory_utilization = utilization_sample.get(
                "gpu_memory_utilization_percent"
            )
            if isinstance(raw_gpu_memory_utilization, (int, float)):
                gpu_memory_utilization_values.append(
                    float(raw_gpu_memory_utilization)
                )
        interval_resources["gpu_utilization_percent_periodic_mean"] = (
            sum(gpu_utilization_values) / len(gpu_utilization_values)
            if gpu_utilization_values
            else None
        )
        interval_resources["gpu_memory_utilization_percent_periodic_mean"] = (
            sum(gpu_memory_utilization_values) / len(gpu_memory_utilization_values)
            if gpu_memory_utilization_values
            else None
        )
        training_wall_seconds = float(interval_resources["wall_seconds"])
        training_resource_measurements = {
            "input_tensor_shapes_first_batch": first_input_shapes,
            "physical_batch_size_graph_days": numeric_distribution(physical_graph_days),
            "effective_batch_size_graph_days": numeric_distribution(effective_graph_days),
            "gradient_accumulation_steps": options.accumulate_steps,
            "data_parallel_workers": 1,
            "stage_host_seconds": stage_host_seconds,
            "stage_cuda_device_seconds": {
                name: _cuda_event_seconds(events)
                for name, events in cuda_stage_events.items()
            },
            "stage_timing_note": (
                "CUDA event times are device durations and may overlap. Host times are "
                "dispatch/wait durations. Collate worker time is summed across workers and "
                "must not be added to wall time. Safety/progress device reads occur only at "
                "10-batch or epoch-end boundaries and are reported as synchronization time."
            ),
            "step_timing": {
                "post_fetch_step_host_seconds": numeric_distribution(
                    post_fetch_step_host_seconds
                ),
                "end_to_end_epoch_wall_seconds": training_wall_seconds,
                "mean_end_to_end_batch_wall_seconds": (
                    training_wall_seconds / len(physical_graph_days)
                    if physical_graph_days
                    else None
                ),
                "note": (
                    "post-fetch timings isolate batch compute/optimizer dispatch; the "
                    "10-batch boundary entries also include their explicitly reported safety "
                    "synchronization; the end-to-end mean includes loader wait, transfer, "
                    "and compute."
                ),
            },
            "resources": interval_resources,
            "steady_cuda_allocated_bytes": numeric_distribution(steady_allocated),
            "steady_cuda_reserved_bytes": numeric_distribution(steady_reserved),
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else None
            ),
            "throughput": {
                "graph_days_per_second": (
                    sum(physical_graph_days) / training_wall_seconds
                    if training_wall_seconds
                    else None
                ),
                "nodes_per_second": (
                    processed_nodes / training_wall_seconds
                    if training_wall_seconds
                    else None
                ),
                "edges_per_second": (
                    processed_edges / training_wall_seconds
                    if training_wall_seconds
                    else None
                ),
            },
        }
        gradient_connectivity = _gradient_connectivity_report(
            model, gradient_connected_parameters
        )
        # Temporal-v7 is the production graph contract and must exercise every
        # registered trainable tensor. Older materialized fixtures can contain
        # a schema route with no usable edge in a particular epoch; retain and
        # report that observed-data gap without misclassifying the parameter as
        # structurally disconnected.
        if (
            temporal_schema
            and not gradient_connectivity[
                "all_trainable_parameters_received_gradient"
            ]
        ):
            raise RuntimeError(
                "trainable parameters are disconnected from the epoch loss: "
                + ", ".join(gradient_connectivity["parameters_without_gradient"])
            )
        validation, _ = _evaluate_model(
            model,
            reusable_validation_loader
            or _loader(
                directory,
                splits["validation"],
                options,
                epoch=epoch,
                training=False,
            ),
            options,
            device,
            dtype,
            planned_rows=validation_plan_rows,
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
            "gradient_connectivity": gradient_connectivity,
            "resource_measurements": training_resource_measurements,
            "validation": validation,
            "elapsed_seconds": time.monotonic() - started,
            "best_epoch": best_epoch,
            "batch_days": options.batch_days,
            "accumulate_steps": options.accumulate_steps,
            "temporal_execution_plan_fingerprint": (
                temporal_execution.plan_fingerprint
                if temporal_execution is not None
                else None
            ),
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
            device=device,
            epoch=epoch + 1,
            global_step=global_step,
            best_score=best_score,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            skipped_optimizer_steps=skipped_optimizer_steps,
            history=history,
            initial_model_state_sha256=initial_model_state_sha256,
            parameter_count=parameter_count,
            trainable_parameter_count=trainable_parameter_count,
            parameter_contract=parameter_contract,
            temporal_execution=temporal_execution,
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
        "resource_inventory": resource_inventory,
        "configuration": asdict(options),
        "model_config": model_config.to_dict(),
        "training_policies": policy,
        "graph_control": _graph_control_report(options),
        "initial_model_state_sha256": initial_model_state_sha256,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "parameter_contract": parameter_contract,
        "all_epochs_trainable_parameters_received_gradient": all(
            row.get("gradient_connectivity", {}).get(
                "all_trainable_parameters_received_gradient"
            )
            is True
            for row in history
        ),
        "temporal_execution": (
            temporal_execution.lineage() if temporal_execution is not None else None
        ),
        "loader_selection": (
            {
                "source": "measured_temporal_cuda_preflight",
                "workers": temporal_execution.loader_workers,
                "prefetch_factor": temporal_execution.loader_prefetch_factor,
                "persistent_workers": temporal_execution.loader_persistent_workers,
                "loader_instances": 2,
                "simultaneous_worker_pools": (
                    2 if temporal_execution.loader_workers > 0 else 0
                ),
                "total_worker_processes": temporal_execution.loader_workers * 2,
                "packed_transfers": True,
                "autotune": dict(temporal_execution.loader_autotune),
            }
            if temporal_execution is not None
            else {
                "source": "explicit_training_configuration",
                "workers": options.workers,
                "prefetch_factor": 2 if options.workers else None,
                "persistent_workers": False,
                "loader_instances": 1,
                "simultaneous_worker_pools": 0,
                "total_worker_processes": options.workers,
                "packed_transfers": True,
            }
        ),
        "physical_execution": {
            "route_edge_chunk_size_configured": options.route_edge_chunk_size,
            "route_edge_chunking_is_lossless": True,
            "architecture_contract": parameter_contract["architecture"],
            "nodes_edges_events_dropped": 0,
        },
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
        "attempted_optimizer_steps": global_step + skipped_optimizer_steps,
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
        "resource_measurements": {
            "preflight": (
                dict(temporal_execution.preflight_resources)
                if temporal_execution is not None
                else None
            ),
            "epochs": [
                {
                    "epoch": row["epoch"],
                    **row["resource_measurements"],
                }
                for row in history
            ],
            "input_tensor_shapes": (
                history[0]["resource_measurements"][
                    "input_tensor_shapes_first_batch"
                ]
                if history
                else None
            ),
            "physical_batch_size_graph_days": (
                history[-1]["resource_measurements"][
                    "physical_batch_size_graph_days"
                ]
                if history
                else None
            ),
            "effective_batch_size_graph_days": (
                history[-1]["resource_measurements"][
                    "effective_batch_size_graph_days"
                ]
                if history
                else None
            ),
        },
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
    temporal_preflight_report: str | Path | None = None,
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
    graph_control = _validate_checkpoint_graph_control(state, options)
    selected, dtype, runtime = _device_and_precision(device, amp)
    directory = Path(dataset_directory or state["dataset_directory"]).expanduser().resolve()
    dataset = open_kbo_graph_dataset(directory)
    if dataset.manifest["fingerprint"] != state["dataset_fingerprint"]:
        raise ValueError("evaluation graph dataset differs from the checkpoint fingerprint")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    temporal_execution: _TemporalExecution | None = None
    if dataset.manifest.get("graph_schema") == "temporal_v7":
        saved_temporal = state.get("temporal_execution")
        if not isinstance(saved_temporal, Mapping):
            raise ValueError("temporal checkpoint is missing its selected CUDA plan")
        selected_report = temporal_preflight_report or saved_temporal.get("report_path")
        if not isinstance(selected_report, (str, Path)):
            raise ValueError("temporal evaluation requires its CUDA preflight report")
        temporal_execution = _load_temporal_execution(
            dataset, options, selected_report, runtime=runtime
        )
        if saved_temporal != temporal_execution.lineage():
            raise ValueError("temporal checkpoint and evaluation execution plans differ")
        if split == "test":
            raise ValueError(
                "held-out temporal test evaluation requires a separately indexed and gated plan"
            )
    elif temporal_preflight_report is not None:
        raise ValueError("legacy graph evaluation does not accept a temporal preflight report")
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
    planned_rows = (
        temporal_execution.split_rows(split) if temporal_execution is not None else None
    )
    metrics, predictions = _evaluate_model(
        model,
        _loader(
            directory,
            days,
            options,
            epoch=0,
            training=False,
            planned_rows=planned_rows,
            workers_override=(
                temporal_execution.loader_workers
                if temporal_execution is not None
                else None
            ),
            prefetch_factor_override=(
                temporal_execution.loader_prefetch_factor
                if temporal_execution is not None
                else None
            ),
            persistent_workers_override=(
                temporal_execution.loader_persistent_workers
                if temporal_execution is not None
                else None
            ),
        ),
        options,
        selected,
        dtype,
        collect_predictions=True,
        planned_rows=planned_rows,
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
        "graph_control": graph_control,
        "temporal_execution": (
            temporal_execution.lineage() if temporal_execution is not None else None
        ),
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
