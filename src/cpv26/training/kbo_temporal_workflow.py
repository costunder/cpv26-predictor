"""Fail-closed single-seed training workflow for temporal-v7 KBO graphs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from cpv26.data.kbo_dataset_loader import open_kbo_graph_dataset
from cpv26.models.kbo_relgnn import KBO_TEMPORAL_ROUTE_NAMES
from cpv26.training import kbo_capacity_comparison as capacity
from cpv26.training import kbo_runner as runner

TEMPORAL_WORKFLOW_PROTOCOL = "temporal_v7_single_seed_full_node_comparison"
TEMPORAL_WORKFLOW_PROTOCOL_VERSION = 1
TEMPORAL_WORKFLOW_REPORT = "temporal_workflow_report.json"
TEMPORAL_PREFLIGHT_REPORT = "temporal_cuda_preflight.json"
TEMPORAL_CHILD_DIRECTORY = "full_node"
TEMPORAL_CAPACITY = {"hidden_dim": 256, "layers": 3, "heads": 8}
TEMPORAL_VARIANTS = ("full", "node_only")
_TRAIN_SEASONS = tuple(range(2001, 2025))
_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def _default_training_config() -> runner.KBOTrainingConfig:
    return runner.KBOTrainingConfig(
        device="cuda:0",
        epochs=30,
        batch_days=8,
        hidden_dim=256,
        layers=3,
        heads=8,
        amp="auto",
        workers=2,
        accumulate_steps=1,
        max_pa_per_day=0,
        max_edges_per_route_per_day=0,
        patience=0,
        seed=2026,
        max_days_per_split=None,
        train_seasons=_TRAIN_SEASONS,
        validation_season=2025,
        test_season=2026,
        chronological=True,
        route_message_normalization="none",
        route_schedule="full",
        graph_control="intact",
        activation_checkpointing=True,
        compact_kbo_channels=True,
    )


@dataclass(frozen=True, slots=True)
class KBOTemporalWorkflowPlan:
    """Paths and runtime options for exactly one matched full/node-only pair."""

    dataset_directory: Path
    output_directory: Path
    config: runner.KBOTrainingConfig = field(default_factory=_default_training_config)
    sample_index_path: Path | None = None
    preflight_report_path: Path | None = None
    max_reserved_fraction: float = 0.85
    seeds: tuple[int, ...] = (2026,)
    variants: tuple[str, ...] = TEMPORAL_VARIANTS

    def __post_init__(self) -> None:
        dataset = Path(self.dataset_directory).expanduser().resolve()
        output = Path(self.output_directory).expanduser().resolve()
        sample_index = (
            dataset / "sample_index.json"
            if self.sample_index_path is None
            else Path(self.sample_index_path).expanduser().resolve()
        )
        preflight_report = (
            output / TEMPORAL_PREFLIGHT_REPORT
            if self.preflight_report_path is None
            else Path(self.preflight_report_path).expanduser().resolve()
        )
        object.__setattr__(self, "dataset_directory", dataset)
        object.__setattr__(self, "output_directory", output)
        object.__setattr__(self, "sample_index_path", sample_index)
        object.__setattr__(self, "preflight_report_path", preflight_report)
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "variants", tuple(self.variants))
        if self.variants != TEMPORAL_VARIANTS:
            raise ValueError("temporal_v7 workflow permits exactly full and node_only")
        if self.seeds != (self.config.seed,):
            raise ValueError("temporal_v7 workflow permits exactly one configured seed")
        _validate_strict_training_config(self.config)
        if output == dataset or output in dataset.parents or dataset in output.parents:
            raise ValueError("temporal archive and workflow output directories must not overlap")
        if sample_index == output or output in sample_index.parents:
            raise ValueError("sample index must not be inside the workflow output directory")
        if sample_index != dataset / "sample_index.json":
            raise ValueError("temporal workflow requires the archive's canonical sample index")
        if preflight_report.parent != output:
            raise ValueError("temporal preflight report must be directly inside workflow output")
        if (
            isinstance(self.max_reserved_fraction, bool)
            or not isinstance(self.max_reserved_fraction, (int, float))
            or not math.isfinite(float(self.max_reserved_fraction))
            or not 0 < float(self.max_reserved_fraction) <= 0.85
        ):
            raise ValueError("temporal max_reserved_fraction must be in (0, 0.85]")


def _validate_strict_training_config(config: runner.KBOTrainingConfig) -> None:
    expected: dict[str, Any] = {
        "epochs": 30,
        "hidden_dim": 256,
        "layers": 3,
        "heads": 8,
        "accumulate_steps": 1,
        "max_pa_per_day": 0,
        "max_edges_per_route_per_day": 0,
        "patience": 0,
        "max_days_per_split": None,
        "train_seasons": _TRAIN_SEASONS,
        "validation_season": 2025,
        "test_season": 2026,
        "chronological": True,
        "route_message_normalization": "none",
        "route_schedule": "full",
        "graph_control": "intact",
        "compact_kbo_channels": True,
        "activation_checkpointing": True,
    }
    differences = [
        f"{name}={getattr(config, name)!r} (expected {value!r})"
        for name, value in expected.items()
        if getattr(config, name) != value
    ]
    if differences:
        raise ValueError(
            "temporal_v7 workflow requires the fixed production contract: "
            + "; ".join(differences)
        )
    if not config.device.startswith("cuda:"):
        raise ValueError("temporal_v7 production workflow requires an explicit CUDA device")


def _load_json(path: Path, *, context: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: {path}")
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _sha256(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _count(value: Any, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return int(value)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plain(value: Any) -> Any:
    """Normalize tuples and mapping subclasses to their persisted JSON form."""

    return json.loads(json.dumps(value, sort_keys=True))


def _quantile(values: Sequence[int], probability: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty topology population")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _quantiles(values: Sequence[int]) -> dict[str, float]:
    return {
        f"p{int(round(probability * 100)):02d}": _quantile(values, probability)
        for probability in _QUANTILES
    }


def _manifest_days(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_days = manifest.get("days")
    if not isinstance(raw_days, list) or not raw_days:
        raise ValueError("temporal_v7 manifest days must be a non-empty list")
    result: dict[str, Mapping[str, Any]] = {}
    for raw in raw_days:
        entry = _mapping(raw, context="temporal_v7 manifest day")
        day_id = entry.get("day")
        if not isinstance(day_id, str):
            raise ValueError("temporal_v7 manifest day is missing its date")
        try:
            parsed = date.fromisoformat(day_id)
        except ValueError as exc:
            raise ValueError(f"temporal_v7 manifest day is invalid: {day_id!r}") from exc
        if parsed.isoformat() != day_id or day_id in result:
            raise ValueError("temporal_v7 manifest days must be unique canonical ISO dates")
        result[day_id] = entry
    return result


def _validate_topology_index(
    manifest: Mapping[str, Any],
    plan: KBOTemporalWorkflowPlan,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    assert plan.sample_index_path is not None
    sample_index, index_sha256 = _load_json(
        plan.sample_index_path, context="temporal_v7 sample index"
    )
    dataset_fingerprint = _sha256(
        manifest.get("fingerprint"), context="temporal_v7 dataset fingerprint"
    )
    if sample_index.get("schema_version") != 2:
        raise ValueError("temporal_v7 sample index schema_version must be 2")
    if sample_index.get("sample_fingerprint_scope") != "all_materialized_arrays_v2":
        raise ValueError("temporal_v7 sample index fingerprint scope differs")
    if sample_index.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("temporal_v7 sample index belongs to a different archive")
    if sample_index.get("sampling_policy_fingerprint") != manifest.get(
        "sampling_policy_fingerprint"
    ):
        raise ValueError("temporal_v7 sample index sampling policy lineage differs")
    if sample_index.get("sampling_policy") != manifest.get("sampling_policy"):
        raise ValueError("temporal_v7 sample index sampling policy differs")
    if sample_index.get("label_year_ceiling") != plan.config.validation_season:
        raise ValueError("temporal_v7 sample index was not built with the validation label ceiling")
    if sample_index.get("held_out_labels_loaded") is not False:
        raise ValueError("temporal_v7 sample index does not prove held-out labels stayed sealed")

    manifest_days = _manifest_days(manifest)
    expected_days = {
        day_id
        for day_id in manifest_days
        if date.fromisoformat(day_id).year
        in {*plan.config.train_seasons, plan.config.validation_season}
    }
    seasons = {date.fromisoformat(day_id).year for day_id in expected_days}
    expected_seasons = {*plan.config.train_seasons, plan.config.validation_season}
    if seasons != expected_seasons:
        missing = sorted(expected_seasons - seasons)
        raise ValueError(f"temporal_v7 archive is missing train/validation seasons: {missing}")

    raw_index_days = sample_index.get("days")
    if not isinstance(raw_index_days, list):
        raise ValueError("temporal_v7 sample index days must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for raw in raw_index_days:
        entry = dict(_mapping(raw, context="temporal_v7 sample-index day"))
        day_id = entry.get("day")
        if not isinstance(day_id, str) or day_id in indexed:
            raise ValueError("temporal_v7 sample index contains an invalid or duplicate day")
        try:
            parsed = date.fromisoformat(day_id)
        except ValueError as exc:
            raise ValueError(f"temporal_v7 sample-index date is invalid: {day_id!r}") from exc
        if parsed.year == plan.config.test_season:
            raise ValueError("temporal_v7 sample index opened a held-out test sample")
        nodes = _mapping(entry.get("sample_nodes"), context=f"{day_id} sample_nodes")
        edges = _mapping(entry.get("sample_edges"), context=f"{day_id} sample_edges")
        if set(nodes) != {"player", "team", "game"}:
            raise ValueError(f"{day_id} sample_nodes must contain player/team/game exactly")
        if set(edges) != set(KBO_TEMPORAL_ROUTE_NAMES):
            raise ValueError(f"{day_id} sample_edges must contain the exact temporal routes")
        entry["sample_nodes"] = {
            name: _count(nodes[name], context=f"{day_id} sample_nodes.{name}", minimum=1)
            for name in ("player", "team", "game")
        }
        entry["sample_edges"] = {
            name: _count(edges[name], context=f"{day_id} sample_edges.{name}")
            for name in KBO_TEMPORAL_ROUTE_NAMES
        }
        entry["sample_fingerprint"] = _sha256(
            entry.get("sample_fingerprint"), context=f"{day_id} sample_fingerprint"
        )
        indexed[day_id] = entry
    if set(indexed) != expected_days:
        missing_days = sorted(expected_days - set(indexed))
        extra_days = sorted(set(indexed) - expected_days)
        raise ValueError(
            "temporal_v7 sample index must cover every and only train/validation day; "
            f"missing={missing_days[:5]}, extra={extra_days[:5]}"
        )
    ordered_entries = [indexed[day_id] for day_id in sorted(indexed)]
    if sample_index.get("fingerprint") != _json_sha256(ordered_entries):
        raise ValueError("temporal_v7 sample index fingerprint does not match its day entries")

    split_rows = {
        "train": [
            indexed[day_id]
            for day_id in sorted(indexed)
            if date.fromisoformat(day_id).year in plan.config.train_seasons
        ],
        "validation": [
            indexed[day_id]
            for day_id in sorted(indexed)
            if date.fromisoformat(day_id).year == plan.config.validation_season
        ],
    }
    split_lineage: dict[str, Any] = {}
    for split, rows in split_rows.items():
        split_lineage[split] = {
            "days": len(rows),
            "date_start": rows[0]["day"],
            "date_end": rows[-1]["day"],
            "ordered_sample_fingerprints_sha256": _json_sha256(
                [(row["day"], row["sample_fingerprint"]) for row in rows]
            ),
            "topology_records_sha256": _json_sha256(rows),
        }
    combined = _json_sha256(
        [(row["day"], row["sample_fingerprint"]) for row in ordered_entries]
    )
    lineage = {
        "sample_index_path": str(plan.sample_index_path),
        "preflight_report_path": str(plan.preflight_report_path),
        "max_reserved_fraction": float(plan.max_reserved_fraction),
        "sample_index_sha256": index_sha256,
        "sample_index_fingerprint": sample_index["fingerprint"],
        "sampling_policy_fingerprint": sample_index["sampling_policy_fingerprint"],
        "splits": split_lineage,
        "combined_train_validation_fingerprint": combined,
        "variant_fingerprints": {variant: combined for variant in TEMPORAL_VARIANTS},
        "all_variants_equal": True,
    }

    def values_for(path: str, name: str | None = None) -> list[int]:
        return [
            int(row[path][name]) if name is not None else sum(row[path].values())
            for row in ordered_entries
        ]

    topology = {
        "sample_count": len(ordered_entries),
        "total_nodes": _quantiles(values_for("sample_nodes")),
        "total_edges": _quantiles(values_for("sample_edges")),
        "nodes_by_type": {
            name: _quantiles(values_for("sample_nodes", name))
            for name in ("player", "team", "game")
        },
        "edges_by_route": {
            name: _quantiles(values_for("sample_edges", name))
            for name in KBO_TEMPORAL_ROUTE_NAMES
        },
    }
    return lineage, topology, index_sha256


def _validate_archive(
    plan: KBOTemporalWorkflowPlan,
) -> tuple[Any, dict[str, Any], dict[str, Any], str, str]:
    if not plan.dataset_directory.is_dir():
        raise FileNotFoundError(f"temporal_v7 archive does not exist: {plan.dataset_directory}")
    manifest_path = plan.dataset_directory / "manifest.json"
    manifest_on_disk, manifest_sha256 = _load_json(
        manifest_path, context="temporal_v7 manifest"
    )
    dataset = open_kbo_graph_dataset(
        plan.dataset_directory,
        label_year_ceiling=plan.config.validation_season,
    )
    manifest = dataset.manifest
    if manifest != manifest_on_disk:
        raise ValueError("opened temporal_v7 archive disagrees with its manifest file")
    if manifest.get("dataset_version") != 7 or manifest.get("graph_schema") != "temporal_v7":
        raise ValueError("workflow requires an exact dataset_version=7 temporal_v7 archive")
    if set(_mapping(manifest.get("node_feature_dims"), context="node_feature_dims")) != {
        "player",
        "team",
        "game",
    }:
        raise ValueError("temporal_v7 archive must declare player/team/game nodes")
    route_dims = _mapping(manifest.get("route_feature_dims"), context="route_feature_dims")
    if set(route_dims) != set(KBO_TEMPORAL_ROUTE_NAMES):
        raise ValueError("temporal_v7 archive does not declare the exact temporal routes")
    for name, value in route_dims.items():
        _count(value, context=f"route_feature_dims.{name}", minimum=1)
    policy = _mapping(manifest.get("archive_policy"), context="archive_policy")
    if policy.get("daily_graph_files") is not False:
        raise ValueError("temporal_v7 workflow rejects materialized daily-snapshot archives")
    topology_lineage, topology, index_sha256 = _validate_topology_index(manifest, plan)
    return dataset, topology_lineage, topology, manifest_sha256, index_sha256


def _validate_training_resources(
    training: Mapping[str, Any], *, variant: str
) -> dict[str, Any]:
    inventory = _mapping(
        training.get("resource_inventory"), context=f"{variant} resource_inventory"
    )
    for name in (
        "logical_cpu_count",
        "allowed_cpu_count",
        "allowed_cpu_ids",
        "physical_ram_bytes",
        "available_ram_bytes_at_start",
        "cgroup_memory_limit_bytes",
        "visible_gpu_count",
        "selected_gpu_name",
        "mig_partition_visible",
    ):
        if name not in inventory:
            raise ValueError(f"temporal child {variant} resource inventory lacks {name}")
    loader = _mapping(
        training.get("loader_selection"), context=f"{variant} loader_selection"
    )
    workers = _count(loader.get("workers"), context=f"{variant} loader workers")
    prefetch = loader.get("prefetch_factor")
    if workers == 0:
        if prefetch is not None or loader.get("persistent_workers") is not False:
            raise ValueError(f"temporal child {variant} zero-worker loader is malformed")
    elif (
        isinstance(prefetch, bool)
        or not isinstance(prefetch, int)
        or prefetch < 1
        or loader.get("persistent_workers") is not True
    ):
        raise ValueError(f"temporal child {variant} parallel loader is malformed")
    if loader.get("source") != "measured_temporal_cuda_preflight":
        raise ValueError(f"temporal child {variant} did not use measured loader tuning")
    expected_pool_contract = {
        "loader_instances": 2,
        "simultaneous_worker_pools": 2 if workers > 0 else 0,
        "total_worker_processes": workers * 2,
    }
    if any(loader.get(name) != value for name, value in expected_pool_contract.items()):
        raise ValueError(
            f"temporal child {variant} loader pool residency differs from preflight"
        )
    autotune = _mapping(loader.get("autotune"), context=f"{variant} loader autotune")
    if autotune.get("status") != "measured":
        raise ValueError(f"temporal child {variant} loader autotune was not measured")
    if training.get("all_epochs_trainable_parameters_received_gradient") is not True:
        raise ValueError(
            f"temporal child {variant} has trainable parameters without epoch gradients"
        )

    resources = _mapping(
        training.get("resource_measurements"), context=f"{variant} resource measurements"
    )
    _mapping(resources.get("preflight"), context=f"{variant} preflight resources")
    epochs = resources.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise ValueError(f"temporal child {variant} has no epoch resource measurements")
    required_epoch_fields = (
        "input_tensor_shapes_first_batch",
        "physical_batch_size_graph_days",
        "effective_batch_size_graph_days",
        "gradient_accumulation_steps",
        "data_parallel_workers",
        "stage_host_seconds",
        "stage_cuda_device_seconds",
        "step_timing",
        "resources",
        "steady_cuda_allocated_bytes",
        "steady_cuda_reserved_bytes",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "throughput",
    )
    for epoch in epochs:
        if not isinstance(epoch, Mapping) or any(
            name not in epoch for name in required_epoch_fields
        ):
            raise ValueError(
                f"temporal child {variant} epoch resource measurements are incomplete"
            )
    physical = _mapping(
        training.get("physical_execution"), context=f"{variant} physical_execution"
    )
    if (
        physical.get("route_edge_chunking_is_lossless") is not True
        or physical.get("nodes_edges_events_dropped") != 0
    ):
        raise ValueError(f"temporal child {variant} physical execution shrank graph semantics")
    return {
        "resource_inventory": dict(inventory),
        "loader_selection": dict(loader),
        "preflight": dict(_mapping(resources["preflight"], context="preflight")),
        "first_epoch": dict(epochs[0]),
        "last_epoch": dict(epochs[-1]),
        "physical_execution": dict(physical),
    }


def _validate_child_runtime(
    child: Mapping[str, Any],
    plan: KBOTemporalWorkflowPlan,
) -> dict[str, Any]:
    raw_runs = _mapping(child.get("runs"), context="temporal child runs")
    if set(raw_runs) != set(TEMPORAL_VARIANTS):
        raise ValueError("temporal child report must contain exactly full and node_only runs")
    child_root = (plan.output_directory / TEMPORAL_CHILD_DIRECTORY).resolve()
    evidence: dict[str, Any] = {}
    reference_runtime: dict[str, Any] | None = None
    for variant in TEMPORAL_VARIANTS:
        run = _mapping(raw_runs[variant], context=f"temporal child {variant}")
        run_directory = Path(str(run.get("run_directory", ""))).expanduser().resolve()
        try:
            run_directory.relative_to(child_root)
        except ValueError as exc:
            raise ValueError(f"temporal child {variant} run escaped its output directory") from exc
        training, training_sha256 = _load_json(
            run_directory / "training_report.json",
            context=f"temporal child {variant} training report",
        )
        if training.get("status") != "completed":
            raise ValueError(f"temporal child {variant} training did not complete")
        if training.get("dataset_fingerprint") != child.get("dataset_fingerprint"):
            raise ValueError(f"temporal child {variant} dataset lineage differs")
        if training.get("held_out_test_season") != plan.config.test_season or training.get(
            "test_used_during_training"
        ) is not False:
            raise ValueError(f"temporal child {variant} opened or mislabeled held-out test")
        runtime = dict(_mapping(training.get("runtime"), context=f"{variant} runtime"))
        if reference_runtime is None:
            reference_runtime = runtime
        elif runtime != reference_runtime:
            raise ValueError("temporal full/node_only child runtimes differ")
        allocated = _count(
            training.get("peak_cuda_allocated_bytes"),
            context=f"{variant} peak_cuda_allocated_bytes",
            minimum=1,
        )
        reserved = _count(
            training.get("peak_cuda_reserved_bytes"),
            context=f"{variant} peak_cuda_reserved_bytes",
            minimum=1,
        )
        if allocated > reserved:
            raise ValueError(f"temporal child {variant} CUDA allocated memory exceeds reserved")
        evidence[variant] = {
            "training_report": str(run_directory / "training_report.json"),
            "training_report_sha256": training_sha256,
            "runtime": runtime,
            "peak_cuda_allocated_bytes": allocated,
            "peak_cuda_reserved_bytes": reserved,
            "resource_execution": _validate_training_resources(
                training, variant=variant
            ),
        }
    assert reference_runtime is not None
    signature = dict(_mapping(child.get("runtime_signature"), context="child runtime_signature"))
    for name, expected in signature.items():
        if reference_runtime.get(name) != expected:
            raise ValueError(f"temporal child runtime_signature.{name} differs from training")
    return {
        "runtime_signature": signature,
        "all_variants_same_runtime": True,
        "variants": evidence,
    }


def _validate_child_report(
    child: Mapping[str, Any],
    plan: KBOTemporalWorkflowPlan,
    *,
    dataset_fingerprint: str,
    topology_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if child.get("status") != "completed":
        raise ValueError("temporal full/node child report is not completed")
    if child.get("protocol") != capacity.FULL_NODE_COMPARISON_PROTOCOL:
        raise ValueError("temporal child report has the wrong protocol")
    if child.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("temporal child report belongs to a different archive")
    if child.get("seed") != plan.config.seed or child.get("variants") != list(
        TEMPORAL_VARIANTS
    ):
        raise ValueError("temporal child report changed its seed or variants")
    if _plain(child.get("training_config")) != _plain(asdict(plan.config)):
        raise ValueError("temporal child report changed the strict training configuration")
    capacity_fields = _mapping(child.get("capacity"), context="temporal child capacity")
    if dict(capacity_fields) != TEMPORAL_CAPACITY:
        raise ValueError("temporal child report changed the fixed model capacity")
    if (
        child.get("held_out_test_season") != plan.config.test_season
        or child.get("selection_split") != "validation"
        or child.get("test_used_for_training_selection_or_comparison") is not False
        or child.get("smoke_test_only") is not False
    ):
        raise ValueError("temporal child report does not keep 2026 test sealed")
    loader = _mapping(child.get("loader_lineage"), context="temporal child loader_lineage")
    if loader.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("temporal child loader archive fingerprint differs")
    variants = _mapping(
        loader.get("variant_fingerprints"), context="child loader variant_fingerprints"
    )
    if set(variants) != set(TEMPORAL_VARIANTS) or len(set(variants.values())) != 1:
        raise ValueError("temporal child loaders do not have identical full/node lineage")
    if loader.get("all_non_route_settings_equal") is not True:
        raise ValueError("temporal child loader settings are not matched")
    comparison = dict(
        _mapping(
            child.get("validation_selection_comparison"),
            context="temporal validation comparison",
        )
    )
    for name in ("full", "node_only", "node_only_minus_full"):
        value = comparison.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"temporal validation comparison {name} must be finite")
    if not math.isclose(
        float(comparison["node_only"]) - float(comparison["full"]),
        float(comparison["node_only_minus_full"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("temporal validation comparison delta is inconsistent")
    comparison["sample_topology_fingerprint"] = topology_fingerprint
    return comparison, _validate_child_runtime(child, plan)


def run_kbo_temporal_workflow(
    plan: KBOTemporalWorkflowPlan,
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Validate topology lineage, then train exactly one full/node-only pair."""

    assert plan.preflight_report_path is not None
    dataset, topology_lineage, topology, manifest_sha, index_sha = _validate_archive(plan)
    dataset_fingerprint = _sha256(
        dataset.manifest.get("fingerprint"), context="temporal_v7 dataset fingerprint"
    )
    plan_payload = {
        "dataset_directory": str(plan.dataset_directory),
        "output_directory": str(plan.output_directory),
        "sample_index_path": str(plan.sample_index_path),
        "preflight_report_path": str(plan.preflight_report_path),
        "max_reserved_fraction": float(plan.max_reserved_fraction),
        "dataset_fingerprint": dataset_fingerprint,
        "archive_manifest_sha256": manifest_sha,
        "sample_index_sha256": index_sha,
        "config": asdict(plan.config),
        "seeds": list(plan.seeds),
        "variants": list(plan.variants),
    }
    plan_fingerprint = _json_sha256(plan_payload)
    report_path = plan.output_directory / TEMPORAL_WORKFLOW_REPORT
    child_directory = plan.output_directory / TEMPORAL_CHILD_DIRECTORY
    if plan.output_directory.exists() and any(plan.output_directory.iterdir()):
        if not report_path.is_file():
            raise FileExistsError(
                "temporal workflow output is non-empty and has no workflow report"
            )
        previous, _ = _load_json(report_path, context="temporal workflow report")
        if previous.get("plan_fingerprint") != plan_fingerprint:
            raise ValueError("temporal workflow resume changed archive, topology, or config")
        unexpected = {
            path.name
            for path in plan.output_directory.iterdir()
            if path.name
            not in {
                TEMPORAL_WORKFLOW_REPORT,
                plan.preflight_report_path.name,
                TEMPORAL_CHILD_DIRECTORY,
            }
        }
        if unexpected:
            raise FileExistsError(
                f"temporal workflow output contains unexpected entries: {sorted(unexpected)}"
            )
    plan.output_directory.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "running",
        "protocol": TEMPORAL_WORKFLOW_PROTOCOL,
        "protocol_version": TEMPORAL_WORKFLOW_PROTOCOL_VERSION,
        "plan_fingerprint": plan_fingerprint,
        "plan": plan_payload,
        "selection_split": "validation",
        "held_out_test": {
            "season": plan.config.test_season,
            "labels_loaded_by_workflow": False,
            "sample_loaded_by_workflow": False,
            "used_for_training_selection_or_comparison": False,
            "sealed": True,
        },
        "topology_size_quantiles": topology,
        "sample_fingerprint_lineage": topology_lineage,
        "child_output_directory": str(child_directory),
        "limitations": [
            "This workflow contains one fixed seed and cannot establish seed stability.",
            "Validation selects checkpoints and compares full versus node_only; 2026 test "
            "labels and samples remain sealed.",
            "node_only retains graph-derived node/role features but registers no relation "
            "layers; shared parameters are identically initialized while variant capacity "
            "is intentionally different.",
            "Each temporal sample uses every legally prior archived record required by "
            "the full-history policy; dynamic batching changes transport boundaries only "
            "and never truncates graph nodes, edges, events, or history.",
            "Raw plate-appearance relation coverage varies by season; player-game aggregate "
            "events provide the cross-era relation contract.",
            "Larger topology and higher CUDA memory use do not by themselves establish "
            "better validation accuracy.",
        ],
    }
    runner._atomic_json(report_path, report)
    failed_stage = "cuda_preflight"
    try:
        progress(
            "temporal_v7 lineage validated; selecting a CUDA execution plan at <="
            f"{float(plan.max_reserved_fraction):.1%} reserved memory"
        )
        from cpv26.training.kbo_temporal_preflight import (
            run_adaptive_temporal_cuda_preflight,
        )

        reuse_preflight = False
        if plan.preflight_report_path.is_file():
            existing_preflight, _ = _load_json(
                plan.preflight_report_path, context="temporal CUDA preflight"
            )
            passed_preflight = (
                existing_preflight.get("status") == "passed"
                and existing_preflight.get("selected_for_training") is True
            )
            if passed_preflight and existing_preflight.get(
                "max_reserved_fraction"
            ) != float(plan.max_reserved_fraction):
                raise ValueError(
                    "passed temporal CUDA preflight memory limit differs from the workflow plan"
                )
            reuse_preflight = passed_preflight
        if reuse_preflight:
            preflight = existing_preflight
            preflight_sha = hashlib.sha256(
                plan.preflight_report_path.read_bytes()
            ).hexdigest()
            _, _, runtime_signature = runner._device_and_precision(
                plan.config.device, plan.config.amp
            )
            runner._load_temporal_execution(
                dataset,
                plan.config,
                plan.preflight_report_path,
                runtime=runtime_signature,
            )
            progress(
                "reusing passed temporal CUDA plan "
                f"{preflight['execution_plan']['plan_fingerprint']}"
            )
        else:
            preflight = run_adaptive_temporal_cuda_preflight(
                dataset,
                plan.config,
                output=plan.preflight_report_path,
                max_reserved_fraction=float(plan.max_reserved_fraction),
                progress=progress,
            )
            _, preflight_sha = _load_json(
                plan.preflight_report_path, context="temporal CUDA preflight"
            )
        selected_plan = _mapping(
            preflight.get("execution_plan"), context="temporal selected execution plan"
        )
        selected_attempt_number = _count(
            preflight.get("selected_attempt"),
            context="temporal selected preflight attempt",
            minimum=1,
        )
        attempts = preflight.get("attempts")
        if (
            not isinstance(attempts, list)
            or selected_attempt_number > len(attempts)
            or not isinstance(attempts[selected_attempt_number - 1], Mapping)
        ):
            raise ValueError("temporal selected preflight attempt evidence is malformed")
        selected_attempt = attempts[selected_attempt_number - 1]
        report["preflight_gate"] = {
            "status": "passed",
            "report": str(plan.preflight_report_path),
            "report_sha256": preflight_sha,
            "selected_attempt": preflight.get("selected_attempt"),
            "plan_fingerprint": selected_plan.get("plan_fingerprint"),
            "budgets": selected_plan.get("budgets"),
            "actual_batch_count": selected_plan.get("actual_batch_count"),
            "peak_reserved_fraction": preflight.get("peak_reserved_fraction"),
            "max_reserved_fraction": preflight.get("max_reserved_fraction"),
            "all_actual_batches_measured": preflight.get(
                "all_actual_batches_measured"
            ),
            "loader_runtime": selected_plan.get("loader_runtime"),
            "physical_batching": selected_plan.get("physical_batching"),
            "loader_autotune": selected_attempt.get("loader_autotune"),
            "resource_measurements": selected_attempt.get("resource_measurements"),
        }
        runner._atomic_json(report_path, report)
        failed_stage = "full_node_training"
        progress(
            "temporal CUDA preflight passed; training exactly seed "
            f"{plan.config.seed} full/node_only"
        )
        child = capacity.train_kbo_full_node_comparison(
            plan.dataset_directory,
            child_directory,
            config=plan.config,
            temporal_preflight_report=plan.preflight_report_path,
            progress=progress,
        )
        # The immutable evidence must not change during a long GPU run.
        _, current_manifest_sha = _load_json(
            plan.dataset_directory / "manifest.json", context="temporal_v7 manifest"
        )
        assert plan.sample_index_path is not None
        _, current_index_sha = _load_json(
            plan.sample_index_path, context="temporal_v7 sample index"
        )
        if (current_manifest_sha, current_index_sha) != (manifest_sha, index_sha):
            raise ValueError("temporal archive or sample index changed during training")
        if hashlib.sha256(plan.preflight_report_path.read_bytes()).hexdigest() != preflight_sha:
            raise ValueError("temporal CUDA preflight report changed during training")
        failed_stage = "final_validation"
        execution_attestation = _mapping(
            child.get("temporal_execution_attestation"),
            context="temporal child execution attestation",
        )
        if (
            execution_attestation.get("all_variants_exact_plan") is not True
            or execution_attestation.get("plan_fingerprint")
            != selected_plan.get("plan_fingerprint")
            or execution_attestation.get("preflight_report_sha256") != preflight_sha
        ):
            raise ValueError("temporal full/node children did not use the gated batch plan")
        comparison, runtime = _validate_child_report(
            child,
            plan,
            dataset_fingerprint=dataset_fingerprint,
            topology_fingerprint=topology_lineage[
                "combined_train_validation_fingerprint"
            ],
        )
        report.update(
            status="completed",
            child_report=str(child_directory / capacity.FULL_NODE_COMPARISON_REPORT),
            child_protocol=child.get("protocol"),
            validation_selection_comparison=comparison,
            gpu_runtime=runtime,
            temporal_execution_attestation=dict(execution_attestation),
        )
        runner._atomic_json(report_path, report)
    except Exception as exc:
        report["status"] = "failed"
        report["failed_stage"] = failed_stage
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        runner._atomic_json(report_path, report)
        raise
    return report


__all__ = [
    "KBOTemporalWorkflowPlan",
    "TEMPORAL_CAPACITY",
    "TEMPORAL_VARIANTS",
    "TEMPORAL_WORKFLOW_PROTOCOL",
    "TEMPORAL_WORKFLOW_PROTOCOL_VERSION",
    "TEMPORAL_WORKFLOW_REPORT",
    "run_kbo_temporal_workflow",
]
