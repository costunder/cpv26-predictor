"""Point-in-time graph snapshots and numeric atomic-route batches."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from .routes import RouteRegistry, default_route_registry


def _as_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _freeze_rows(
    values: Iterable[Sequence[float]],
    *,
    expected_rows: int | None = None,
    field_name: str,
) -> tuple[tuple[float, ...], ...]:
    rows = tuple(tuple(float(item) for item in row) for row in values)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"{field_name} has {len(rows)} rows; expected {expected_rows}")
    widths = {len(row) for row in rows}
    if len(widths) > 1:
        raise ValueError(f"{field_name} must be rectangular")
    return rows


@dataclass(frozen=True, slots=True)
class TorchAtomicRouteBatch:
    """Torch tensor view of an :class:`AtomicRouteBatch`.

    The type intentionally stores tensors as ``Any`` so importing the graph
    package does not import PyTorch.
    """

    route_name: str
    source_type: str
    destination_type: str
    source_index: Any
    destination_index: Any
    event_features: Any
    event_age_seconds: Any
    publication_delay_seconds: Any
    weights: Any
    bidirectional: bool

    @property
    def num_edges(self) -> int:
        return int(self.source_index.numel())

    @property
    def age_seconds(self) -> Any:
        """Compatibility alias for event age at the prediction cutoff."""

        return self.event_age_seconds

    def pin_memory(self) -> TorchAtomicRouteBatch:
        """Let PyTorch's DataLoader pin nested route tensors for asynchronous CUDA copies."""
        return replace(
            self,
            source_index=self.source_index.pin_memory(),
            destination_index=self.destination_index.pin_memory(),
            event_features=self.event_features.pin_memory(),
            event_age_seconds=self.event_age_seconds.pin_memory(),
            publication_delay_seconds=self.publication_delay_seconds.pin_memory(),
            weights=self.weights.pin_memory(),
        )


@dataclass(frozen=True, slots=True)
class AtomicRouteBatch:
    """Observed two-hop events for one whitelisted atomic route."""

    route_name: str
    source_type: str
    destination_type: str
    source_index: tuple[int, ...]
    destination_index: tuple[int, ...]
    event_at: tuple[datetime, ...]
    available_at: tuple[datetime, ...]
    event_features: tuple[tuple[float, ...], ...] = ()
    weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        source_index = tuple(int(value) for value in self.source_index)
        destination_index = tuple(int(value) for value in self.destination_index)
        event_at = tuple(_as_utc(value, "event_at") for value in self.event_at)
        available_at = tuple(_as_utc(value, "available_at") for value in self.available_at)
        count = len(source_index)
        if len(destination_index) != count or len(event_at) != count or len(available_at) != count:
            raise ValueError(
                "source_index, destination_index, event_at, and available_at must have equal length"
            )
        if any(index < 0 for index in source_index + destination_index):
            raise ValueError("route indices must be non-negative")

        raw_features = self.event_features
        event_features: tuple[tuple[float, ...], ...]
        if not raw_features and count:
            event_features = tuple(() for _ in range(count))
        else:
            event_features = _freeze_rows(
                raw_features,
                expected_rows=count,
                field_name="event_features",
            )
        weights = tuple(float(value) for value in self.weights)
        if not weights:
            weights = (1.0,) * count
        if len(weights) != count:
            raise ValueError("weights must be empty or have one value per route event")
        if any(value < 0.0 for value in weights):
            raise ValueError("route weights must be non-negative")

        object.__setattr__(self, "source_index", source_index)
        object.__setattr__(self, "destination_index", destination_index)
        object.__setattr__(self, "event_at", event_at)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "event_features", event_features)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def from_columns(
        cls,
        *,
        route_name: str,
        source_type: str,
        destination_type: str,
        source_index: Iterable[int],
        destination_index: Iterable[int],
        event_at: Iterable[datetime],
        available_at: Iterable[datetime],
        event_features: Iterable[Sequence[float]] = (),
        weights: Iterable[float] = (),
    ) -> AtomicRouteBatch:
        return cls(
            route_name=route_name,
            source_type=source_type,
            destination_type=destination_type,
            source_index=tuple(source_index),
            destination_index=tuple(destination_index),
            event_at=tuple(event_at),
            available_at=tuple(available_at),
            event_features=tuple(tuple(row) for row in event_features),
            weights=tuple(weights),
        )

    @property
    def num_edges(self) -> int:
        return len(self.source_index)

    @property
    def feature_dim(self) -> int:
        return len(self.event_features[0]) if self.event_features else 0

    def event_ages_seconds(self, cutoff_at: datetime) -> tuple[float, ...]:
        """Return signed ages relative to the event occurrence/effective time."""

        cutoff_utc = _as_utc(cutoff_at, "cutoff_at")
        return tuple((cutoff_utc - timestamp).total_seconds() for timestamp in self.event_at)

    @property
    def publication_delays_seconds(self) -> tuple[float, ...]:
        """Return signed publication delay, ``available_at - event_at``."""

        return tuple(
            (available - event).total_seconds()
            for event, available in zip(self.event_at, self.available_at, strict=True)
        )

    def temporal_features(
        self,
        cutoff_at: datetime,
        *,
        include_publication_delay: bool = False,
    ) -> tuple[tuple[float, ...], ...]:
        """Expose raw temporal seconds for feature pipelines.

        Event age is always included. Publication delay is opt-in because it
        can encode source-specific reporting behavior that should be audited.
        """

        ages = self.event_ages_seconds(cutoff_at)
        if not include_publication_delay:
            return tuple((age,) for age in ages)
        return tuple(
            (age, delay) for age, delay in zip(ages, self.publication_delays_seconds, strict=True)
        )

    def validate(
        self,
        *,
        cutoff_at: datetime,
        node_counts: Mapping[str, int],
        registry: RouteRegistry,
    ) -> None:
        cutoff_utc = _as_utc(cutoff_at, "cutoff_at")
        registry.validate_endpoints(
            self.route_name,
            self.source_type,
            self.destination_type,
        )
        if any(timestamp > cutoff_utc for timestamp in self.available_at):
            first_future = next(
                timestamp for timestamp in self.available_at if timestamp > cutoff_utc
            )
            raise ValueError(
                f"route {self.route_name!r} contains information available at "
                f"{first_future.isoformat()}, after cutoff {cutoff_utc.isoformat()}"
            )
        if self.source_type not in node_counts:
            raise ValueError(f"unknown source node type: {self.source_type}")
        if self.destination_type not in node_counts:
            raise ValueError(f"unknown destination node type: {self.destination_type}")
        if self.source_index and max(self.source_index) >= node_counts[self.source_type]:
            raise IndexError(f"source index exceeds {self.source_type!r} node count")
        if (
            self.destination_index
            and max(self.destination_index) >= node_counts[self.destination_type]
        ):
            raise IndexError(f"destination index exceeds {self.destination_type!r} node count")

    def as_torch(
        self,
        *,
        cutoff_at: datetime,
        registry: RouteRegistry | None = None,
        device: Any = None,
        dtype: Any = None,
    ) -> TorchAtomicRouteBatch:
        """Convert this batch to tensors without making torch an import dependency."""

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to tensorize graph batches; install the project's "
                "ml dependency group"
            ) from exc

        cutoff_utc = _as_utc(cutoff_at, "cutoff_at")
        route_registry = registry or default_route_registry()
        route = route_registry.validate_endpoints(
            self.route_name,
            self.source_type,
            self.destination_type,
        )
        if any(timestamp > cutoff_utc for timestamp in self.available_at):
            raise ValueError(
                f"route {self.route_name!r} contains information newer than its cutoff"
            )
        float_dtype = dtype or torch.float32
        event_features = torch.tensor(
            self.event_features,
            dtype=float_dtype,
            device=device,
        )
        if event_features.ndim == 1:
            event_features = event_features.reshape(self.num_edges, 0)
        event_age_seconds = torch.tensor(
            self.event_ages_seconds(cutoff_utc),
            dtype=float_dtype,
            device=device,
        )
        publication_delay_seconds = torch.tensor(
            self.publication_delays_seconds,
            dtype=float_dtype,
            device=device,
        )
        return TorchAtomicRouteBatch(
            route_name=self.route_name,
            source_type=self.source_type,
            destination_type=self.destination_type,
            source_index=torch.tensor(self.source_index, dtype=torch.long, device=device),
            destination_index=torch.tensor(
                self.destination_index,
                dtype=torch.long,
                device=device,
            ),
            event_features=event_features,
            event_age_seconds=event_age_seconds,
            publication_delay_seconds=publication_delay_seconds,
            weights=torch.tensor(self.weights, dtype=float_dtype, device=device),
            bidirectional=route.bidirectional,
        )


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """Leakage-resistant graph state as known at one prediction cutoff."""

    snapshot_id: str
    cutoff_at: datetime
    node_ids: Mapping[str, tuple[Hashable, ...]]
    node_features: Mapping[str, tuple[tuple[float, ...], ...]]
    routes: tuple[AtomicRouteBatch, ...]
    registry: RouteRegistry = field(default_factory=default_route_registry, repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    node_feature_dims: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.snapshot_id or self.snapshot_id.strip() != self.snapshot_id:
            raise ValueError("snapshot_id must be a non-empty, trimmed string")
        cutoff_at = _as_utc(self.cutoff_at, "cutoff_at")

        frozen_ids: dict[str, tuple[Hashable, ...]] = {}
        frozen_features: dict[str, tuple[tuple[float, ...], ...]] = {}
        explicit_dims = {
            node_type: int(width) for node_type, width in self.node_feature_dims.items()
        }
        if any(width < 0 for width in explicit_dims.values()):
            raise ValueError("node feature dimensions must be non-negative")
        for node_type, identifiers in self.node_ids.items():
            ids = tuple(identifiers)
            if len(set(ids)) != len(ids):
                raise ValueError(f"node IDs for {node_type!r} must be unique")
            frozen_ids[node_type] = ids
            if node_type not in self.node_features:
                raise ValueError(f"missing feature matrix for node type {node_type!r}")
            rows = _freeze_rows(
                self.node_features[node_type],
                expected_rows=len(ids),
                field_name=f"node_features[{node_type!r}]",
            )
            if rows:
                actual_dim = len(rows[0])
                if node_type in explicit_dims and explicit_dims[node_type] != actual_dim:
                    raise ValueError(
                        f"node feature dimension for {node_type!r} is "
                        f"{actual_dim}; declared {explicit_dims[node_type]}"
                    )
                explicit_dims[node_type] = actual_dim
            elif node_type not in explicit_dims:
                raise ValueError(
                    f"empty node type {node_type!r} requires an explicit feature dimension"
                )
            frozen_features[node_type] = rows
        unexpected_features = set(self.node_features).difference(frozen_ids)
        if unexpected_features:
            names = ", ".join(sorted(unexpected_features))
            raise ValueError(f"features supplied for unknown node types: {names}")
        unexpected_dims = set(explicit_dims).difference(frozen_ids)
        if unexpected_dims:
            names = ", ".join(sorted(unexpected_dims))
            raise ValueError(f"feature dimensions supplied for unknown node types: {names}")

        routes = tuple(self.routes)
        node_counts = {key: len(value) for key, value in frozen_ids.items()}
        for route_batch in routes:
            route_batch.validate(
                cutoff_at=cutoff_at,
                node_counts=node_counts,
                registry=self.registry,
            )

        object.__setattr__(self, "cutoff_at", cutoff_at)
        object.__setattr__(self, "node_ids", MappingProxyType(frozen_ids))
        object.__setattr__(self, "node_features", MappingProxyType(frozen_features))
        object.__setattr__(self, "routes", routes)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "node_feature_dims", MappingProxyType(explicit_dims))

    @property
    def node_counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {node_type: len(identifiers) for node_type, identifiers in self.node_ids.items()}
        )

    @property
    def feature_dims(self) -> Mapping[str, int]:
        return self.node_feature_dims

    def node_index(self, node_type: str) -> Mapping[Hashable, int]:
        try:
            identifiers = self.node_ids[node_type]
        except KeyError as exc:
            raise KeyError(f"unknown node type: {node_type}") from exc
        return MappingProxyType({identifier: index for index, identifier in enumerate(identifiers)})

    def torch_node_features(self, *, device: Any = None, dtype: Any = None) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to tensorize graph snapshots; install the "
                "project's ml dependency group"
            ) from exc
        float_dtype = dtype or torch.float32
        tensors: dict[str, Any] = {}
        for node_type, rows in self.node_features.items():
            tensor = torch.tensor(rows, dtype=float_dtype, device=device)
            tensors[node_type] = tensor.reshape(
                len(rows),
                self.node_feature_dims[node_type],
            )
        return tensors

    def torch_routes(
        self,
        *,
        device: Any = None,
        dtype: Any = None,
    ) -> tuple[TorchAtomicRouteBatch, ...]:
        return tuple(
            batch.as_torch(
                cutoff_at=self.cutoff_at,
                registry=self.registry,
                device=device,
                dtype=dtype,
            )
            for batch in self.routes
        )
