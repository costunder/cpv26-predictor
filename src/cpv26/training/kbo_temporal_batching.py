"""Deterministic node/edge-budget minibatches for temporal KBO samples."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TemporalSampleSize:
    day: date
    nodes: int
    edges: int
    fingerprint: str

    def __post_init__(self) -> None:
        if self.nodes < 1 or self.edges < 1:
            raise ValueError("temporal sample sizes must be positive")
        if len(self.fingerprint) != 64:
            raise ValueError("temporal sample fingerprint must be SHA-256")


def load_temporal_sample_sizes(
    directory: str | Path,
    *,
    dataset_fingerprint: str,
    sampling_policy_fingerprint: str,
) -> dict[date, TemporalSampleSize]:
    """Load the validation-bounded sample index used to plan device batches."""

    root = Path(directory).expanduser().resolve()
    path = root / "sample_index.json"
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError("temporal sample index must contain a JSON object")
    if report.get("schema_version") != 2:
        raise ValueError("temporal sample index schema_version must be 2")
    if report.get("sample_fingerprint_scope") != "all_materialized_arrays_v2":
        raise ValueError("temporal sample index fingerprint scope differs")
    if report.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("temporal sample index belongs to a different archive")
    if report.get("sampling_policy_fingerprint") != sampling_policy_fingerprint:
        raise ValueError("temporal sample index uses a different sampling policy")
    rows = report.get("days")
    if not isinstance(rows, list) or not rows:
        raise ValueError("temporal sample index has no indexed days")
    result: dict[date, TemporalSampleSize] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("temporal sample index day is malformed")
        raw_nodes, raw_edges = row.get("sample_nodes"), row.get("sample_edges")
        if not isinstance(raw_nodes, Mapping) or not isinstance(raw_edges, Mapping):
            raise ValueError("temporal sample index lacks typed node/edge counts")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*raw_nodes.values(), *raw_edges.values())
        ):
            raise ValueError("temporal sample index counts must be non-negative integers")
        sample = TemporalSampleSize(
            day=date.fromisoformat(str(row["day"])),
            nodes=sum(int(value) for value in raw_nodes.values()),
            edges=sum(int(value) for value in raw_edges.values()),
            fingerprint=str(row["sample_fingerprint"]),
        )
        if sample.day in result:
            raise ValueError("temporal sample index contains a duplicate day")
        result[sample.day] = sample
    encoded = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if report.get("fingerprint") != hashlib.sha256(encoded.encode("utf-8")).hexdigest():
        raise ValueError("temporal sample index fingerprint is inconsistent")
    return result


class TemporalBudgetBatchSampler:
    """First-fit consecutive packing with deterministic optional shuffling.

    A single oversize day is yielded by itself so no observation disappears.
    All other batches obey every configured budget.  The yielded values are
    indices into the caller's ``days`` sequence, as expected by DataLoader.
    """

    def __init__(
        self,
        days: Sequence[date],
        sizes: Mapping[date, TemporalSampleSize],
        *,
        max_nodes: int,
        max_edges: int,
        max_days: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        for name, value in (
            ("max_nodes", max_nodes),
            ("max_edges", max_edges),
            ("max_days", max_days),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        missing = [day for day in days if day not in sizes]
        if missing:
            raise ValueError(
                "temporal sample index is missing selected days: "
                + ", ".join(day.isoformat() for day in missing[:5])
            )
        self.days = tuple(days)
        self.sizes = sizes
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_days = max_days
        self.shuffle = shuffle
        self.seed = seed
        self._batches = self._pack()

    def _pack(self) -> tuple[tuple[int, ...], ...]:
        order = list(range(len(self.days)))
        if self.shuffle:
            random.Random(self.seed).shuffle(order)
        batches: list[tuple[int, ...]] = []
        current: list[int] = []
        nodes = 0
        edges = 0
        for index in order:
            size = self.sizes[self.days[index]]
            would_overflow = bool(current) and (
                len(current) >= self.max_days
                or nodes + size.nodes > self.max_nodes
                or edges + size.edges > self.max_edges
            )
            if would_overflow:
                batches.append(tuple(current))
                current, nodes, edges = [], 0, 0
            current.append(index)
            nodes += size.nodes
            edges += size.edges
            if size.nodes > self.max_nodes or size.edges > self.max_edges:
                batches.append(tuple(current))
                current, nodes, edges = [], 0, 0
        if current:
            batches.append(tuple(current))
        return tuple(batches)

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self._batches:
            yield list(batch)

    def __len__(self) -> int:
        return len(self._batches)


__all__ = [
    "TemporalBudgetBatchSampler",
    "TemporalSampleSize",
    "load_temporal_sample_sizes",
]
