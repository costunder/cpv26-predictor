"""Open the immutable KBO graph representation declared by its manifest."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from .kbo_graph_dataset import GraphDay, KBOGraphDataset


class KBOGraphDatasetLike(Protocol):
    """Small interface shared by materialized snapshots and temporal samples."""

    directory: Path
    manifest: dict[str, Any]

    def days(self) -> tuple[date, ...]: ...

    def load_day(self, day: date | str) -> GraphDay: ...


def open_kbo_graph_dataset(
    directory: str | Path,
    *,
    label_year_ceiling: int | None = None,
) -> KBOGraphDatasetLike:
    """Dispatch without silently treating a temporal archive as a v5 cache.

    ``label_year_ceiling`` is also the raw temporal-shard decode ceiling. This
    keeps held-out source records sealed inside persistent DataLoader workers,
    not merely absent from their requested day list.
    """

    resolved = Path(directory).expanduser().resolve()
    manifest_path = resolved / "manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("KBO graph manifest must contain a JSON object")
    version = manifest.get("dataset_version")
    schema = manifest.get("graph_schema")
    if version == 7 or schema == "temporal_v7":
        if version != 7 or schema != "temporal_v7":
            raise ValueError("temporal KBO manifest version and graph_schema disagree")
        from .kbo_temporal_archive import KBOTemporalGraphDataset

        return KBOTemporalGraphDataset(
            resolved,
            label_year_ceiling=label_year_ceiling,
        )
    # Materialized v2-v5 datasets already store one immutable graph per day.
    # Their held-out labels/records are not decoded while loading a requested
    # training day, so the temporal raw-shard ceiling has nothing to enforce.
    # Keep accepting the argument because the shared runner must work across
    # every supported archive generation.
    return KBOGraphDataset(resolved)


__all__ = ["KBOGraphDatasetLike", "open_kbo_graph_dataset"]
