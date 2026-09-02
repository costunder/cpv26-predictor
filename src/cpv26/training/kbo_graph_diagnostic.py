"""Same-checkpoint diagnostics for KBO RelGNN graph dependence.

The transforms in this module deliberately operate on an already validated,
CPU-side disjoint batch.  They never mutate a graph cache, checkpoint, model,
or input batch.  Endpoint randomisation is keyed by the stable day identifier,
not the minibatch position, so changing ``batch_days`` cannot move an endpoint
to another day or silently change the experiment.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from cpv26.graph import TorchAtomicRouteBatch
from cpv26.models._torch import require_torch
from cpv26.models.kbo_relgnn import (
    KBO_ROUTE_NAMES,
    KBO_VNEXT_ROUTE_NAMES,
    KBORelGNNConfig,
    KBORelGNNModel,
)
from cpv26.models.relgnn import RelGNNDiagnosticsObserver

_ROUTE_TENSOR_FIELDS = (
    "source_index",
    "destination_index",
    "event_features",
    "event_age_seconds",
    "publication_delay_seconds",
    "weights",
)
_ATTRIBUTE_FIELDS = (
    "event_features",
    "event_age_seconds",
    "publication_delay_seconds",
    "weights",
)
_TRANSFORM_MODES = {
    "intact",
    "no_routes",
    "permute_endpoints",
    "permute_edge_attributes",
    "route_knockout",
}


@dataclass(frozen=True, slots=True)
class KBOGraphTransformSpec:
    """One immutable inference-time graph intervention."""

    mode: str = "intact"
    seed: int = 2026
    route_name: str | None = None
    reviewed_route_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.mode not in _TRANSFORM_MODES:
            raise ValueError(f"unsupported graph transform mode: {self.mode}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("graph transform seed must be a non-negative integer")
        reviewed: tuple[str, ...] = KBO_ROUTE_NAMES
        if self.reviewed_route_names is not None:
            if not isinstance(self.reviewed_route_names, tuple):
                raise TypeError("reviewed_route_names must be a tuple when supplied")
            reviewed = _validated_route_names(self.reviewed_route_names)
            object.__setattr__(self, "reviewed_route_names", reviewed)
        if self.mode == "route_knockout":
            if self.route_name not in reviewed:
                raise ValueError("route knockout requires one reviewed KBO route")
        elif self.route_name is not None:
            raise ValueError("route_name is valid only for route_knockout")


def _validated_route_names(route_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(route_names)
    if not names:
        raise ValueError("diagnostic route names cannot be empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("diagnostic route names must be non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("diagnostic route names must be unique")
    return names


def _checkpoint_route_names(state: Mapping[str, Any]) -> tuple[str, ...]:
    """Read the exact route contract saved by this checkpoint.

    Checkpoints predating ``route_feature_dims`` used only the four legacy KBO
    routes, so that genuinely old format retains its historical diagnostic
    condition set.  Modern legacy and vNext checkpoints take their names from
    their own model config rather than a process-global route union.
    """

    model_config = state.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint model_config must be a mapping")
    raw_route_dims = model_config.get("route_feature_dims")
    if raw_route_dims is None:
        return KBO_ROUTE_NAMES
    if not isinstance(raw_route_dims, Mapping):
        raise ValueError("checkpoint route_feature_dims must be a mapping")
    saved = _validated_route_names(tuple(raw_route_dims))
    # JSON manifests are written with sorted keys, which must not reorder the
    # established diagnostic conditions.  The checkpoint still controls the
    # exact route *set*; this only restores the reviewed semantic order.
    reviewed = tuple(name for name in KBO_VNEXT_ROUTE_NAMES if name in saved)
    unknown = tuple(name for name in saved if name not in KBO_VNEXT_ROUTE_NAMES)
    return (*reviewed, *unknown)


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _non_identity_permutation(count: int, *key: object) -> np.ndarray[Any, Any]:
    if count < 0:
        raise ValueError("permutation size cannot be negative")
    if count < 2:
        return np.arange(count, dtype=np.int64)
    result = (
        np.random.default_rng(_stable_seed(*key)).permutation(count).astype(np.int64, copy=False)
    )
    if np.array_equal(result, np.arange(count, dtype=np.int64)):
        result = np.roll(result, 1)
    return result


def _require_cpu_batch(batch: Mapping[str, Any]) -> None:
    if batch.get("_validated_on_cpu") is not True:
        raise ValueError("graph diagnostics require a collated, CPU-validated KBO batch")
    for route in batch.get("routes", ()):
        if not isinstance(route, TorchAtomicRouteBatch):
            raise TypeError("routes must contain TorchAtomicRouteBatch values")
        for field in _ROUTE_TENSOR_FIELDS:
            value = getattr(route, field)
            if value.device.type != "cpu":
                raise ValueError("graph transforms must run before the batch is moved off CPU")
    node_graph_index = batch.get("node_graph_index")
    if not isinstance(node_graph_index, Mapping):
        raise ValueError("batch is missing node_graph_index")
    if any(value.device.type != "cpu" for value in node_graph_index.values()):
        raise ValueError("node graph indices must remain on CPU during transformation")


def _route_edge_days(
    batch: Mapping[str, Any], route: TorchAtomicRouteBatch
) -> tuple[Any, Any, Any]:
    torch, _ = require_torch()
    node_graph_index = batch["node_graph_index"]
    source_graph = node_graph_index[route.source_type][route.source_index]
    destination_graph = node_graph_index[route.destination_type][route.destination_index]
    if not torch.equal(source_graph, destination_graph):
        raise ValueError(f"route {route.route_name!r} contains a cross-day edge")
    day_ids = tuple(str(value) for value in batch["day_ids"])
    if source_graph.numel() and (
        int(source_graph.min()) < 0 or int(source_graph.max()) >= len(day_ids)
    ):
        raise ValueError("route edge refers to an unknown day graph")
    return source_graph, destination_graph, day_ids


def _empty_route(route: TorchAtomicRouteBatch) -> TorchAtomicRouteBatch:
    return replace(
        route,
        **{field: getattr(route, field)[:0] for field in _ROUTE_TENSOR_FIELDS},
    )


def _permuted_endpoint(
    endpoint: Any,
    *,
    node_membership: Any,
    edge_membership: Any,
    day_ids: Sequence[str],
    seed: int,
    route_name: str,
    side: str,
) -> tuple[Any, int]:
    torch, _ = require_torch()
    result = endpoint.clone()
    changes = 0
    for raw_graph_index in torch.unique(edge_membership, sorted=True).tolist():
        graph_index = int(raw_graph_index)
        edge_positions = torch.nonzero(edge_membership == graph_index, as_tuple=False).flatten()
        node_positions = torch.nonzero(node_membership == graph_index, as_tuple=False).flatten()
        if not int(node_positions.numel()):
            raise ValueError("edge graph contains no compatible endpoint nodes")
        permutation = _non_identity_permutation(
            int(node_positions.numel()), seed, day_ids[graph_index], route_name, side
        )
        remapped_nodes = node_positions[torch.from_numpy(permutation)]
        lookup = torch.full(
            (int(node_membership.numel()),), -1, dtype=torch.long, device=endpoint.device
        )
        lookup[node_positions] = remapped_nodes
        previous = result[edge_positions]
        remapped = lookup[previous]
        if bool((remapped < 0).any()):
            raise ValueError("route endpoint does not belong to its declared day")
        result[edge_positions] = remapped
        changes += int((previous != remapped).sum().item())
    return result, changes


def _permute_route_endpoints(
    batch: Mapping[str, Any], route: TorchAtomicRouteBatch, seed: int
) -> tuple[TorchAtomicRouteBatch, int, int]:
    source_graph, destination_graph, day_ids = _route_edge_days(batch, route)
    memberships = batch["node_graph_index"]
    source, source_changes = _permuted_endpoint(
        route.source_index,
        node_membership=memberships[route.source_type],
        edge_membership=source_graph,
        day_ids=day_ids,
        seed=seed,
        route_name=route.route_name,
        side="source",
    )
    destination, destination_changes = _permuted_endpoint(
        route.destination_index,
        node_membership=memberships[route.destination_type],
        edge_membership=destination_graph,
        day_ids=day_ids,
        seed=seed,
        route_name=route.route_name,
        side="destination",
    )
    transformed = replace(route, source_index=source, destination_index=destination)
    # Validate the new endpoints independently of the construction above.  This
    # assertion catches future changes to disjoint-union collation semantics.
    _route_edge_days(batch, transformed)
    return transformed, source_changes, destination_changes


def _permute_route_attributes(
    batch: Mapping[str, Any], route: TorchAtomicRouteBatch, seed: int
) -> tuple[TorchAtomicRouteBatch, int]:
    torch, _ = require_torch()
    source_graph, _, day_ids = _route_edge_days(batch, route)
    columns = {field: getattr(route, field).clone() for field in _ATTRIBUTE_FIELDS}
    changed_rows = 0
    for raw_graph_index in torch.unique(source_graph, sorted=True).tolist():
        graph_index = int(raw_graph_index)
        positions = torch.nonzero(source_graph == graph_index, as_tuple=False).flatten()
        permutation = _non_identity_permutation(
            int(positions.numel()), seed, day_ids[graph_index], route.route_name, "attributes"
        )
        order = positions[torch.from_numpy(permutation)]
        for field in _ATTRIBUTE_FIELDS:
            columns[field][positions] = getattr(route, field)[order]
        changed_rows += int(np.count_nonzero(permutation != np.arange(len(permutation))))
    return replace(route, **columns), changed_rows


def transform_kbo_graph_batch(
    batch: Mapping[str, Any], spec: KBOGraphTransformSpec
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a transformed shallow batch copy and a JSON-safe transformation audit."""

    _require_cpu_batch(batch)
    routes = tuple(batch.get("routes", ()))
    route_names = tuple(route.route_name for route in routes)
    if len(route_names) != len(set(route_names)):
        raise ValueError("diagnostic batch contains duplicate route names")
    if spec.mode == "route_knockout" and spec.route_name not in route_names:
        raise ValueError(f"diagnostic batch is missing route {spec.route_name!r}")

    transformed_routes: list[TorchAtomicRouteBatch] = []
    per_route: dict[str, dict[str, int]] = {}
    for route in routes:
        _route_edge_days(batch, route)
        before = route.num_edges
        source_changes = destination_changes = attribute_changes = 0
        if spec.mode == "no_routes" or (
            spec.mode == "route_knockout" and route.route_name == spec.route_name
        ):
            transformed = _empty_route(route)
        elif spec.mode == "permute_endpoints":
            transformed, source_changes, destination_changes = _permute_route_endpoints(
                batch, route, spec.seed
            )
        elif spec.mode == "permute_edge_attributes":
            transformed, attribute_changes = _permute_route_attributes(batch, route, spec.seed)
        else:
            transformed = route
        transformed_routes.append(transformed)
        per_route[route.route_name] = {
            "edges_before": before,
            "edges_after": transformed.num_edges,
            "edges_removed": before - transformed.num_edges,
            "source_endpoints_changed": source_changes,
            "destination_endpoints_changed": destination_changes,
            "edge_attribute_rows_permuted": attribute_changes,
        }

    result = dict(batch)
    result["routes"] = tuple(transformed_routes)
    totals = {
        key: sum(values[key] for values in per_route.values())
        for key in next(iter(per_route.values()), {})
    }
    effective_changes = (
        totals.get("edges_removed", 0)
        + totals.get("source_endpoints_changed", 0)
        + totals.get("destination_endpoints_changed", 0)
        + totals.get("edge_attribute_rows_permuted", 0)
    )
    audit: dict[str, Any] = {
        "mode": spec.mode,
        "seed": spec.seed,
        "route_name": spec.route_name,
        "batches": 1,
        "days": len(tuple(batch["day_ids"])),
        "day_ids": list(map(str, batch["day_ids"])),
        **totals,
        "effective_changes": effective_changes,
        "no_op": effective_changes == 0,
        "per_route": per_route,
    }
    return result, audit


class KBOGraphBatchTransform:
    """Callable batch transform that accumulates audits across an evaluation pass."""

    def __init__(self, spec: KBOGraphTransformSpec) -> None:
        self.spec = spec
        self._batches = 0
        self._days: set[str] = set()
        self._totals: dict[str, int] = {}
        self._routes: dict[str, dict[str, int]] = {}

    def __call__(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        transformed, audit = transform_kbo_graph_batch(batch, self.spec)
        self._batches += 1
        self._days.update(audit["day_ids"])
        for key, value in audit.items():
            is_counter = (
                key in {"batches", "days", "effective_changes"}
                or key.startswith("edges_")
                or key.endswith(("_changed", "_permuted"))
            )
            if is_counter and isinstance(value, int):
                self._totals[key] = self._totals.get(key, 0) + value
        for route_name, values in audit["per_route"].items():
            target = self._routes.setdefault(route_name, {})
            for key, value in values.items():
                target[key] = target.get(key, 0) + int(value)
        return transformed

    def report(self) -> dict[str, Any]:
        effective_changes = sum(
            self._totals.get(key, 0)
            for key in (
                "edges_removed",
                "source_endpoints_changed",
                "destination_endpoints_changed",
                "edge_attribute_rows_permuted",
            )
        )
        return {
            "mode": self.spec.mode,
            "seed": self.spec.seed,
            "route_name": self.spec.route_name,
            "batches": self._batches,
            "days": len(self._days),
            "day_ids": sorted(self._days),
            **{
                key: value
                for key, value in self._totals.items()
                if key not in {"batches", "days", "effective_changes"}
            },
            "effective_changes": effective_changes,
            "no_op": effective_changes == 0,
            "per_route": self._routes,
        }


_NOT_NUMERIC = object()


def _numeric_delta(reference: Any, candidate: Any) -> Any:
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        result: dict[str, Any] = {}
        for key in reference.keys() & candidate.keys():
            value = _numeric_delta(reference[key], candidate[key])
            if value is not _NOT_NUMERIC:
                result[str(key)] = value
        return result
    if reference is None or candidate is None:
        return None
    if (
        isinstance(reference, Real)
        and not isinstance(reference, bool)
        and isinstance(candidate, Real)
        and not isinstance(candidate, bool)
    ):
        left, right = float(reference), float(candidate)
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("metric deltas require finite numeric values")
        return right - left
    return _NOT_NUMERIC


def recursive_numeric_metric_deltas(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Return candidate-minus-reference deltas while retaining metric nesting."""

    result = _numeric_delta(reference, candidate)
    if not isinstance(result, dict):
        raise TypeError("metric reports must be mappings")
    return result


def _rows_by_query(rows: Sequence[Mapping[str, Any]], task: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if "query_id" not in row:
            raise ValueError(f"{task} prediction row is missing query_id")
        query_id = str(row["query_id"])
        if query_id in indexed:
            raise ValueError(f"duplicate {task} prediction query_id: {query_id}")
        indexed[query_id] = row
    return indexed


def _numbered_column_key(name: str) -> tuple[str, int | str]:
    prefix, _, suffix = name.rpartition("_")
    return prefix, int(suffix) if suffix.isdigit() else suffix


def _finite_vector(
    row: Mapping[str, Any], columns: Sequence[str], task: str
) -> np.ndarray[Any, Any]:
    values = np.asarray([row[column] for column in columns], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{task} predictions contain non-finite values")
    return values


def paired_prediction_sensitivity(
    reference: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Compare predictions after an exact inner join on immutable query identifiers."""

    if set(reference) != set(candidate):
        raise ValueError("prediction task sets differ between diagnostic conditions")
    report: dict[str, Any] = {}
    for task in sorted(reference):
        left = _rows_by_query(reference[task], task)
        right = _rows_by_query(candidate[task], task)
        if set(left) != set(right):
            missing = sorted(set(left) - set(right))
            extra = sorted(set(right) - set(left))
            raise ValueError(
                f"{task} prediction populations differ; missing={missing[:3]}, extra={extra[:3]}"
            )
        query_ids = sorted(left)
        probability_columns: list[str] = []
        scalar_columns: list[str] = []
        if query_ids:
            shared_columns = set(left[query_ids[0]]) & set(right[query_ids[0]])
            probability_columns = sorted(
                (name for name in shared_columns if name.startswith("probability_")),
                key=_numbered_column_key,
            )
            scalar_columns = sorted(
                name for name in shared_columns if name.startswith(("expected_", "predicted_"))
            )
        probability_changes: list[np.ndarray[Any, Any]] = []
        total_variations: list[float] = []
        argmax_flips = 0
        scalar_changes: dict[str, list[float]] = {name: [] for name in scalar_columns}
        for query_id in query_ids:
            left_row, right_row = left[query_id], right[query_id]
            evidence_columns = {
                name
                for name in set(left_row) & set(right_row)
                if name == "label" or name.startswith("observed_")
            }
            if any(left_row[name] != right_row[name] for name in evidence_columns):
                raise ValueError(f"{task} observed label changed for query {query_id}")
            current_probability_columns = sorted(
                (
                    name
                    for name in set(left_row) & set(right_row)
                    if name.startswith("probability_")
                ),
                key=_numbered_column_key,
            )
            if current_probability_columns != probability_columns:
                raise ValueError(f"{task} prediction probability schema differs by query")
            if probability_columns:
                left_values = _finite_vector(left_row, probability_columns, task)
                right_values = _finite_vector(right_row, probability_columns, task)
                change = right_values - left_values
                probability_changes.append(change)
                total_variations.append(float(np.abs(change).sum() * 0.5))
                argmax_flips += int(np.argmax(left_values) != np.argmax(right_values))
            for column in scalar_columns:
                values = _finite_vector(left_row, (column,), task)
                changed = _finite_vector(right_row, (column,), task)
                scalar_changes[column].append(float(changed[0] - values[0]))

        if probability_changes:
            changes = np.stack(probability_changes)
            absolute = np.abs(changes)
            task_report: dict[str, Any] = {
                "queries": len(query_ids),
                "probability_columns": probability_columns,
                "mean_total_variation": float(np.mean(total_variations)),
                "median_total_variation": float(np.median(total_variations)),
                "p95_total_variation": float(np.quantile(total_variations, 0.95)),
                "max_total_variation": float(np.max(total_variations)),
                "mean_absolute_probability_change": float(absolute.mean()),
                "max_absolute_probability_change": float(absolute.max()),
                "root_mean_squared_probability_change": float(np.sqrt(np.mean(changes**2))),
                "argmax_flip_count": argmax_flips,
                "argmax_flip_rate": argmax_flips / len(query_ids),
            }
        else:
            task_report = {
                "queries": len(query_ids),
                "probability_columns": [],
                "mean_total_variation": None,
                "median_total_variation": None,
                "p95_total_variation": None,
                "max_total_variation": None,
                "mean_absolute_probability_change": None,
                "max_absolute_probability_change": None,
                "root_mean_squared_probability_change": None,
                "argmax_flip_count": None,
                "argmax_flip_rate": None,
            }
        task_report["scalar_outputs"] = {
            column: {
                "mean_absolute_change": float(np.mean(np.abs(values))) if values else None,
                "max_absolute_change": float(np.max(np.abs(values))) if values else None,
                "root_mean_squared_change": float(np.sqrt(np.mean(np.square(values))))
                if values
                else None,
            }
            for column, values in scalar_changes.items()
        }
        report[task] = task_report
    return report


@dataclass(slots=True)
class _RunningMoments:
    count: int = 0
    total: float = 0.0
    square_total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, values: Any) -> None:
        torch, _ = require_torch()
        tensor = torch.as_tensor(values).detach().float().reshape(-1)
        if not int(tensor.numel()):
            return
        if not bool(torch.isfinite(tensor).all().item()):
            raise FloatingPointError("RelGNN diagnostic observer received non-finite values")
        self.count += int(tensor.numel())
        self.total += float(tensor.sum().item())
        self.square_total += float(tensor.square().sum().item())
        self.minimum = min(self.minimum, float(tensor.min().item()))
        self.maximum = max(self.maximum, float(tensor.max().item()))

    def report(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.total / self.count
        variance = max(0.0, self.square_total / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


def _moments(bucket: dict[str, Any], name: str) -> _RunningMoments:
    value = bucket.get(name)
    if value is None:
        value = _RunningMoments()
        bucket[name] = value
    return value


def _report_bucket(bucket: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.report() if isinstance(value, _RunningMoments) else value
        for key, value in bucket.items()
    }


class RelGNNDiagnosticsCollector(RelGNNDiagnosticsObserver):
    """Online observer for topology, attention, route gates, and state updates.

    Attention and route-gate weights that are mathematically forced to one are
    reported separately from genuinely competitive normalisations.  Mixing
    those populations would make a high mean weight look like learned focus.
    """

    def __init__(self) -> None:
        self._batches = 0
        self._days: set[str] = set()
        self._nodes: dict[str, int] = {}
        self._queries: dict[str, int] = {}
        self._topology_routes: dict[str, dict[str, int]] = {}
        self._attention: dict[str, dict[str, Any]] = {}
        self._gates: dict[str, dict[str, Any]] = {}
        self._gate_routes: dict[str, dict[str, Any]] = {}

    def begin_batch(self, batch: Mapping[str, Any]) -> None:
        _require_cpu_batch(batch)
        torch, _ = require_torch()
        self._batches += 1
        self._days.update(map(str, batch["day_ids"]))
        for kind, values in batch["node_features"].items():
            self._nodes[kind] = self._nodes.get(kind, 0) + int(values.shape[0])
        for task in ("match", "live_hit", "pa", "box_pa", "box_pitch"):
            values = batch.get(f"{task}_query_ids", ())
            self._queries[task] = self._queries.get(task, 0) + len(values)
        for route in batch["routes"]:
            source_graph, _, _ = _route_edge_days(batch, route)
            target = self._topology_routes.setdefault(route.route_name, {})
            positive = route.weights > 0
            target["edges"] = target.get("edges", 0) + route.num_edges
            target["positive_weight_edges"] = target.get("positive_weight_edges", 0) + int(
                positive.sum().item()
            )
            target["zero_weight_edges"] = target.get("zero_weight_edges", 0) + int(
                (~positive).sum().item()
            )
            for graph_index in torch.unique(source_graph, sorted=True).tolist():
                positions = torch.nonzero(
                    source_graph == int(graph_index), as_tuple=False
                ).flatten()
                selected = positions[positive[positions]]
                if not int(selected.numel()):
                    continue
                destination = route.destination_index[selected]
                _, degree = torch.unique(destination, return_counts=True)
                target["reached_destination_occurrences"] = target.get(
                    "reached_destination_occurrences", 0
                ) + int(degree.numel())
                target["forced_singleton_destinations"] = target.get(
                    "forced_singleton_destinations", 0
                ) + int((degree == 1).sum().item())
                target["competitive_destinations"] = target.get(
                    "competitive_destinations", 0
                ) + int((degree > 1).sum().item())

    def observe_attention(self, **event: Any) -> None:
        torch, _ = require_torch()
        required = {
            "layer_index",
            "route_name",
            "direction",
            "destination_index",
            "positive_weight",
            "attention",
            "message",
            "route_mask",
            "destination_state",
        }
        missing = required - event.keys()
        if missing:
            raise ValueError(f"attention diagnostic event is missing {sorted(missing)}")
        key = f"layer_{event['layer_index']}|{event['route_name']}|{event['direction']}"
        bucket = self._attention.setdefault(
            key,
            {
                "layer_index": int(event["layer_index"]),
                "route_name": str(event["route_name"]),
                "direction": str(event["direction"]),
                "source_channel": str(event.get("source_channel", "")),
                "destination_channel": str(event.get("destination_channel", "")),
                "calls": 0,
                "positive_edges": 0,
                "forced_singleton_destinations": 0,
                "competitive_destinations": 0,
            },
        )
        bucket["calls"] += 1
        destination_index = event["destination_index"].detach().long()
        positive = event["positive_weight"].detach().bool()
        attention = event["attention"].detach().float()
        if attention.ndim != 2 or int(attention.shape[0]) != int(destination_index.numel()):
            raise ValueError("attention observer expects an [edges, heads] tensor")
        if int(positive.numel()) != int(destination_index.numel()):
            raise ValueError("attention positive-weight mask does not match edge count")
        bucket["positive_edges"] += int(positive.sum().item())
        destination_count = int(event["destination_state"].shape[0])
        degree = torch.zeros(destination_count, dtype=torch.long, device=destination_index.device)
        if bool(positive.any()):
            degree.index_add_(
                0,
                destination_index[positive],
                torch.ones_like(destination_index[positive]),
            )
        forced_destinations = degree == 1
        competitive_destinations = degree > 1
        bucket["forced_singleton_destinations"] += int(forced_destinations.sum().item())
        bucket["competitive_destinations"] += int(competitive_destinations.sum().item())
        if int(destination_index.numel()):
            edge_degree = degree[destination_index]
            forced_edges = positive & (edge_degree == 1)
            competitive_edges = positive & (edge_degree > 1)
            _moments(bucket, "forced_singleton_attention").add(attention[forced_edges])
            _moments(bucket, "competitive_attention").add(attention[competitive_edges])
        if bool(positive.any()):
            probabilities = attention[positive].clamp_min(torch.finfo(attention.dtype).tiny)
            positive_destination = destination_index[positive]
            entropy = torch.zeros(
                (destination_count, int(attention.shape[1])),
                dtype=attention.dtype,
                device=attention.device,
            )
            entropy.index_add_(0, positive_destination, -probabilities * probabilities.log())
            maximum = torch.full_like(entropy, -torch.inf)
            maximum.scatter_reduce_(
                0,
                positive_destination[:, None].expand_as(probabilities),
                probabilities,
                reduce="amax",
                include_self=True,
            )
            if bool(competitive_destinations.any()):
                normalizer = degree[competitive_destinations].float().log()[:, None]
                _moments(bucket, "competitive_normalized_entropy").add(
                    entropy[competitive_destinations] / normalizer
                )
                _moments(bucket, "competitive_max_attention").add(maximum[competitive_destinations])
        route_mask = event["route_mask"].detach().bool()
        if bool(route_mask.any()):
            message_norm = event["message"].detach().float().norm(dim=-1)[route_mask]
            state_norm = event["destination_state"].detach().float().norm(dim=-1)[route_mask]
            _moments(bucket, "message_norm").add(message_norm)
            _moments(bucket, "message_to_destination_state_norm").add(
                message_norm / state_norm.clamp_min(1e-12)
            )

    def observe_gates(self, **event: Any) -> None:
        torch, _ = require_torch()
        required = {
            "layer_index",
            "destination_channel",
            "route_names",
            "directions",
            "gate_keys",
            "message_normalization",
            "pre_normalization_messages",
            "messages",
            "masks",
            "route_attention",
            "previous_state",
            "combined_message",
            "candidate_state",
            "updated_state",
        }
        missing = required - event.keys()
        if missing:
            raise ValueError(f"gate diagnostic event is missing {sorted(missing)}")
        message_normalization = str(event["message_normalization"])
        if message_normalization not in {"none", "layer_norm"}:
            raise ValueError("gate diagnostic message_normalization is unsupported")
        key = f"layer_{event['layer_index']}|{event['destination_channel']}"
        bucket = self._gates.setdefault(
            key,
            {
                "layer_index": int(event["layer_index"]),
                "destination_channel": str(event["destination_channel"]),
                "message_normalization": message_normalization,
                "calls": 0,
                "nodes_with_messages": 0,
                "forced_singleton_nodes": 0,
                "competitive_nodes": 0,
            },
        )
        if bucket["message_normalization"] != message_normalization:
            raise ValueError("gate diagnostic events mix message normalization policies")
        bucket["calls"] += 1
        masks = event["masks"].detach().bool()
        weights = event["route_attention"].detach().float()
        if masks.shape != weights.shape or masks.ndim != 2:
            raise ValueError("route gate masks and attention must share shape [nodes, routes]")
        pre_normalization_messages = event["pre_normalization_messages"].detach().float()
        gate_input_messages = event["messages"].detach().float()
        if (
            gate_input_messages.ndim != 3
            or pre_normalization_messages.ndim != 3
            or tuple(gate_input_messages.shape[:2]) != tuple(masks.shape)
            or tuple(pre_normalization_messages.shape) != tuple(gate_input_messages.shape)
        ):
            raise ValueError(
                "pre-normalization and gate-input messages must share shape "
                "[nodes, routes, hidden]"
            )
        active_count = masks.sum(dim=1)
        any_message = active_count > 0
        forced_nodes = active_count == 1
        competitive_nodes = active_count > 1
        bucket["nodes_with_messages"] += int(any_message.sum().item())
        bucket["forced_singleton_nodes"] += int(forced_nodes.sum().item())
        bucket["competitive_nodes"] += int(competitive_nodes.sum().item())
        _moments(bucket, "forced_singleton_gate_weight").add(weights[masks & forced_nodes[:, None]])
        _moments(bucket, "competitive_gate_weight").add(weights[masks & competitive_nodes[:, None]])
        if bool(competitive_nodes.any()):
            safe = weights.clamp_min(torch.finfo(weights.dtype).tiny)
            entropy = -(weights * safe.log()).sum(dim=1)
            _moments(bucket, "competitive_normalized_gate_entropy").add(
                entropy[competitive_nodes] / active_count[competitive_nodes].float().log()
            )
            _moments(bucket, "competitive_max_gate_weight").add(
                weights[competitive_nodes].max(dim=1).values
            )
        route_names = tuple(map(str, event["route_names"]))
        directions = tuple(map(str, event["directions"]))
        source_channels = tuple(map(str, event.get("source_channels", ("",) * len(route_names))))
        if not (len(route_names) == len(directions) == len(source_channels) == masks.shape[1]):
            raise ValueError("route gate metadata does not match route dimension")
        for index, (route_name, direction, source_channel) in enumerate(
            zip(route_names, directions, source_channels, strict=True)
        ):
            route_key = (
                f"layer_{event['layer_index']}|{event['destination_channel']}|"
                f"{route_name}|{direction}"
            )
            target = self._gate_routes.setdefault(
                route_key,
                {
                    "layer_index": int(event["layer_index"]),
                    "destination_channel": str(event["destination_channel"]),
                    "source_channel": source_channel,
                    "route_name": route_name,
                    "direction": direction,
                    "message_normalization": message_normalization,
                    "active_nodes": 0,
                    "competitive_nodes": 0,
                },
            )
            target["active_nodes"] += int(masks[:, index].sum().item())
            target["competitive_nodes"] += int((masks[:, index] & competitive_nodes).sum().item())
            _moments(target, "gate_weight").add(weights[masks[:, index], index])
            _moments(target, "competitive_gate_weight").add(
                weights[masks[:, index] & competitive_nodes, index]
            )
            active = masks[:, index]
            pre_normalization_norm = pre_normalization_messages[:, index].norm(dim=-1)
            gate_input_norm = gate_input_messages[:, index].norm(dim=-1)
            gate_weighted_input_norm = (
                gate_input_messages[:, index] * weights[:, index, None]
            ).norm(dim=-1)
            _moments(target, "pre_normalization_message_norm").add(
                pre_normalization_norm[active]
            )
            _moments(target, "gate_input_message_norm").add(gate_input_norm[active])
            _moments(target, "gate_weighted_input_message_norm").add(
                gate_weighted_input_norm[active]
            )
            if bool(competitive_nodes.any()):
                winners = weights.argmax(dim=1) == index
                target["competitive_winner_count"] = target.get(
                    "competitive_winner_count", 0
                ) + int((winners & masks[:, index] & competitive_nodes).sum().item())

        if bool(any_message.any()):
            previous = event["previous_state"].detach().float()
            combined = event["combined_message"].detach().float()
            candidate = event["candidate_state"].detach().float()
            updated = event["updated_state"].detach().float()
            previous_norm = previous.norm(dim=-1)
            update_norm = (updated - previous).norm(dim=-1)
            candidate_change = (candidate - previous).norm(dim=-1)
            combined_norm = combined.norm(dim=-1)
            _moments(bucket, "previous_state_norm").add(previous_norm[any_message])
            _moments(bucket, "combined_message_norm").add(combined_norm[any_message])
            _moments(bucket, "candidate_state_change_norm").add(candidate_change[any_message])
            _moments(bucket, "actual_update_norm").add(update_norm[any_message])
            _moments(bucket, "relative_update_norm").add(
                update_norm[any_message] / previous_norm[any_message].clamp_min(1e-12)
            )
            _moments(bucket, "forced_singleton_update_norm").add(update_norm[forced_nodes])
            _moments(bucket, "competitive_update_norm").add(update_norm[competitive_nodes])
            state_denominator = previous_norm * updated.norm(dim=-1)
            valid_state_cosine = any_message & (state_denominator > 1e-12)
            if bool(valid_state_cosine.any()):
                _moments(bucket, "previous_to_updated_cosine").add(
                    (previous * updated).sum(dim=-1)[valid_state_cosine]
                    / state_denominator[valid_state_cosine]
                )
            update = updated - previous
            alignment_denominator = combined_norm * update_norm
            valid_alignment = any_message & (alignment_denominator > 1e-12)
            if bool(valid_alignment.any()):
                _moments(bucket, "message_to_update_cosine").add(
                    (combined * update).sum(dim=-1)[valid_alignment]
                    / alignment_denominator[valid_alignment]
                )

    def report(self) -> dict[str, Any]:
        topology_routes: dict[str, Any] = {}
        for key, value in sorted(self._topology_routes.items()):
            topology_item: dict[str, Any] = dict(value)
            reached = int(value.get("reached_destination_occurrences", 0))
            topology_item["forced_singleton_fraction_among_reached"] = (
                int(value.get("forced_singleton_destinations", 0)) / reached
                if reached
                else None
            )
            topology_item["competitive_fraction_among_reached"] = (
                int(value.get("competitive_destinations", 0)) / reached if reached else None
            )
            topology_routes[key] = topology_item
        attention_routes: dict[str, Any] = {}
        for key, value in sorted(self._attention.items()):
            attention_item = _report_bucket(value)
            reached = int(value["forced_singleton_destinations"]) + int(
                value["competitive_destinations"]
            )
            attention_item["forced_singleton_fraction_among_reached"] = (
                int(value["forced_singleton_destinations"]) / reached if reached else None
            )
            attention_item["competitive_fraction_among_reached"] = (
                int(value["competitive_destinations"]) / reached if reached else None
            )
            attention_routes[key] = attention_item
        gate_channels: dict[str, Any] = {}
        for key, value in sorted(self._gates.items()):
            gate_item = _report_bucket(value)
            active = int(value["nodes_with_messages"])
            gate_item["forced_singleton_fraction"] = (
                int(value["forced_singleton_nodes"]) / active if active else None
            )
            gate_item["competitive_fraction"] = (
                int(value["competitive_nodes"]) / active if active else None
            )
            gate_channels[key] = gate_item
        gate_routes: dict[str, Any] = {}
        for key, value in sorted(self._gate_routes.items()):
            route_item = _report_bucket(value)
            # Schema-v1 readers used these shorter names.  Keep them as exact
            # report aliases while schema v2 makes each position around the
            # optional normalization boundary explicit.
            route_item["raw_message_norm"] = dict(
                route_item["pre_normalization_message_norm"]
            )
            route_item["gate_weighted_message_norm"] = dict(
                route_item["gate_weighted_input_message_norm"]
            )
            competitive = int(value["competitive_nodes"])
            route_item["competitive_winner_fraction"] = (
                int(value.get("competitive_winner_count", 0)) / competitive if competitive else None
            )
            gate_routes[key] = route_item
        return {
            "schema_version": 2,
            "batches": self._batches,
            "days": len(self._days),
            "topology": {
                "node_occurrences": dict(sorted(self._nodes.items())),
                "query_counts": dict(sorted(self._queries.items())),
                "routes": topology_routes,
            },
            "attention": {
                "forced_singleton_definition": (
                    "one positive-weight incoming edge for a destination; attention is forced to 1"
                ),
                "by_layer_route_direction": attention_routes,
            },
            "route_gates": {
                "forced_singleton_definition": (
                    "one active incoming route for a node; route-gate weight is forced to 1"
                ),
                "message_norm_definitions": {
                    "pre_normalization_message_norm": (
                        "L2 norm of the route aggregate before optional message normalization"
                    ),
                    "gate_input_message_norm": (
                        "L2 norm of the actual message supplied to the route gate and combiner"
                    ),
                    "gate_weighted_input_message_norm": (
                        "L2 norm after multiplying the actual gate-input message by its "
                        "route-gate weight"
                    ),
                },
                "compatibility_aliases": {
                    "raw_message_norm": "pre_normalization_message_norm",
                    "gate_weighted_message_norm": "gate_weighted_input_message_norm",
                },
                "by_layer_channel": gate_channels,
                "by_route_direction": gate_routes,
            },
        }


def _condition_specs(
    seed: int,
    include_edge_attribute_permutation: bool,
    route_names: Sequence[str] = KBO_ROUTE_NAMES,
) -> tuple[tuple[str, KBOGraphTransformSpec], ...]:
    reviewed_route_names = _validated_route_names(route_names)
    result = [
        ("intact", KBOGraphTransformSpec("intact", seed)),
        ("no_routes", KBOGraphTransformSpec("no_routes", seed)),
        ("permuted_endpoints", KBOGraphTransformSpec("permute_endpoints", seed)),
    ]
    if include_edge_attribute_permutation:
        result.append(
            (
                "permuted_edge_attributes",
                KBOGraphTransformSpec("permute_edge_attributes", seed),
            )
        )
    result.extend(
        (
            f"without_{route_name}",
            KBOGraphTransformSpec(
                "route_knockout",
                seed,
                route_name,
                reviewed_route_names=reviewed_route_names,
            ),
        )
        for route_name in reviewed_route_names
    )
    return tuple(result)


def diagnose_kbo_graph_dependence(
    checkpoint: str | Path,
    *,
    dataset_directory: str | Path | None = None,
    split: str = "validation",
    device: str = "cuda:0",
    amp: str = "auto",
    batch_days: int = 2,
    workers: int = 2,
    seed: int = 2026,
    max_days: int | None = None,
    output_directory: str | Path | None = None,
    include_edge_attribute_permutation: bool = True,
) -> dict[str, Any]:
    """Run same-checkpoint graph interventions and write one paired JSON report.

    This establishes whether the selected checkpoint *depends* on its graph at
    inference time.  It is not a substitute for retraining graph-free and
    topology-randomised baselines with multiple seeds.
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("diagnostic seed must be a non-negative integer")
    if max_days is not None and (
        isinstance(max_days, bool) or not isinstance(max_days, int) or max_days < 1
    ):
        raise ValueError("max_days must be a positive integer when supplied")
    if not isinstance(include_edge_attribute_permutation, bool):
        raise TypeError("include_edge_attribute_permutation must be boolean")

    # Imported lazily to keep the transformation and report helpers usable in
    # lightweight tests without creating an import cycle through kbo_runner.
    from cpv26.data.kbo_graph_dataset import KBOGraphDataset
    from cpv26.data.kbo_playbyplay import sha256_file
    from cpv26.training import kbo_runner as runner

    torch, _ = require_torch()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    state = runner._read_checkpoint(checkpoint_path)
    options = runner.KBOTrainingConfig.from_dict(state["training_config"])
    options = replace(
        options,
        device=device,
        amp=amp,
        batch_days=batch_days,
        workers=workers,
        seed=seed,
        max_days_per_split=max_days if max_days is not None else options.max_days_per_split,
    )
    checkpoint_graph_control = runner._validate_checkpoint_graph_control(state, options)
    route_names = _checkpoint_route_names(state)
    selected, dtype, runtime = runner._device_and_precision(device, amp)
    directory = Path(dataset_directory or state["dataset_directory"]).expanduser().resolve()
    dataset = KBOGraphDataset(directory)
    if dataset.manifest["fingerprint"] != state["dataset_fingerprint"]:
        raise ValueError("diagnostic graph dataset differs from the checkpoint fingerprint")
    days = runner._split_days(dataset, options)[split]
    if not days:
        raise ValueError(f"no dates available for the requested {split} split")
    model: Any = KBORelGNNModel(KBORelGNNConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    model.to(selected)
    model.eval()
    if selected.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    else:
        torch.cuda.reset_peak_memory_stats(selected)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    output = (
        (
            Path(output_directory)
            if output_directory is not None
            else checkpoint_path.parent / "diagnostics" / f"graph-dependence-{run_id}"
        )
        .expanduser()
        .resolve()
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("diagnostic output directory is not empty; use a new directory")
    output.mkdir(parents=True, exist_ok=True)

    baseline_metrics: dict[str, Any] | None = None
    baseline_predictions: dict[str, list[dict[str, Any]]] | None = None
    conditions: dict[str, Any] = {}
    for name, spec in _condition_specs(
        seed, include_edge_attribute_permutation, route_names
    ):
        transform = KBOGraphBatchTransform(spec)
        # Full tensor statistics force device-to-host synchronisation.  One
        # intact pass is sufficient to inspect learned attention/gates; the
        # intervention passes retain paired metrics and prediction deltas.
        collector = RelGNNDiagnosticsCollector() if name == "intact" else None
        metrics, predictions = runner._evaluate_model(
            model,
            runner._loader(directory, days, options, epoch=0, training=False),
            options,
            selected,
            dtype,
            collect_predictions=True,
            batch_transform=transform,
            diagnostics=collector,
        )
        if baseline_metrics is None:
            baseline_metrics = metrics
            baseline_predictions = predictions
        assert baseline_predictions is not None and baseline_metrics is not None
        conditions[name] = {
            "transform": transform.report(),
            "metrics": metrics,
            "metric_delta_vs_intact": recursive_numeric_metric_deltas(baseline_metrics, metrics),
            "prediction_sensitivity_vs_intact": paired_prediction_sensitivity(
                baseline_predictions, predictions
            ),
            "internal_diagnostics": collector.report() if collector is not None else None,
        }

    report: dict[str, Any] = {
        "protocol": "same_checkpoint_inference_graph_ablation",
        "model": "role_aware_composite_relgnn",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(state["epoch"]),
        "checkpoint_graph_control": checkpoint_graph_control,
        "route_names": list(route_names),
        "dataset_directory": str(directory),
        "dataset_fingerprint": dataset.manifest["fingerprint"],
        "split": split,
        "seed": seed,
        "date_start": min(days).isoformat(),
        "date_end": max(days).isoformat(),
        "days": len(days),
        "smoke_test_only": options.max_days_per_split is not None,
        "max_days_per_split": options.max_days_per_split,
        "runtime": runtime,
        "prediction_sensitivity_definitions": {
            "match": "total variation over the match class probability distribution",
            "live_hit": (
                "total variation over the binary hit/no-hit marginal; this is not the full "
                "joint PA-by-hit distribution"
            ),
            "pa": "total variation over the plate-appearance outcome class distribution",
        },
        "internal_diagnostics_scope": "intact condition only",
        "conditions": conditions,
        "output_directory": str(output),
        "limitations": [
            (
                "Same-checkpoint interventions measure current reliance and introduce "
                "distribution shift."
            ),
            "They do not measure the value of a graph model retrained from scratch.",
            "Singleton attention and singleton route gates are reported separately because their "
            "weights are forced to one.",
            (
                "Internal attention, gate, message, and state-update statistics are collected only "
                "for the intact condition to avoid synchronising every intervention pass."
            ),
            (
                "Small deltas should be checked against repeated intact evaluations and multiple "
                "intervention seeds before they are treated as graph effects."
            ),
            "CUDA scatter atomics may prevent bit-for-bit repetition even with a fixed seed.",
        ],
        **runner._runtime_memory(selected),
    }
    runner._atomic_json(output / "report.json", report)
    return report


__all__ = [
    "KBOGraphBatchTransform",
    "KBOGraphTransformSpec",
    "RelGNNDiagnosticsCollector",
    "diagnose_kbo_graph_dependence",
    "paired_prediction_sensitivity",
    "recursive_numeric_metric_deltas",
    "transform_kbo_graph_batch",
]
