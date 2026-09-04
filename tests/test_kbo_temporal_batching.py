from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cpv26.training import kbo_runner
from cpv26.training.kbo_temporal_batching import (
    TemporalBudgetBatchSampler,
    TemporalSampleSize,
    load_temporal_sample_sizes,
)


def _size(day: int, nodes: int, edges: int) -> TemporalSampleSize:
    return TemporalSampleSize(
        date(2025, 4, day), nodes, edges, hashlib.sha256(str(day).encode()).hexdigest()
    )


def test_temporal_budget_sampler_packs_without_dropping_oversize_day() -> None:
    samples = [_size(1, 30, 40), _size(2, 40, 50), _size(3, 120, 250), _size(4, 10, 10)]
    sizes = {sample.day: sample for sample in samples}
    sampler = TemporalBudgetBatchSampler(
        [sample.day for sample in samples],
        sizes,
        max_nodes=100,
        max_edges=100,
        max_days=8,
        shuffle=False,
        seed=2026,
    )
    assert list(sampler) == [[0, 1], [2], [3]]
    assert len(sampler) == 3


def test_temporal_budget_sampler_shuffle_is_seeded_and_complete() -> None:
    samples = [_size(day, 10, 10) for day in range(1, 9)]
    sizes = {sample.day: sample for sample in samples}

    def packed(seed: int) -> list[list[int]]:
        return list(
            TemporalBudgetBatchSampler(
                [sample.day for sample in samples],
                sizes,
                max_nodes=30,
                max_edges=30,
                max_days=3,
                shuffle=True,
                seed=seed,
            )
        )

    assert packed(7) == packed(7)
    assert packed(7) != packed(8)
    assert sorted(index for batch in packed(7) for index in batch) == list(range(8))


def test_temporal_sample_index_checks_lineage_and_content(tmp_path: Path) -> None:
    row = {
        "day": "2025-04-01",
        "sample_nodes": {"player": 10, "team": 2, "game": 5},
        "sample_edges": {"team_game_event": 10, "batter_game_event": 20},
        "sample_fingerprint": "a" * 64,
    }
    payload = {
        "schema_version": 2,
        "sample_fingerprint_scope": "all_materialized_arrays_v2",
        "dataset_fingerprint": "dataset",
        "sampling_policy_fingerprint": "policy",
        "days": [row],
    }
    encoded = json.dumps(payload["days"], sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = hashlib.sha256(encoded.encode()).hexdigest()
    (tmp_path / "sample_index.json").write_text(json.dumps(payload), encoding="utf-8")

    result = load_temporal_sample_sizes(
        tmp_path,
        dataset_fingerprint="dataset",
        sampling_policy_fingerprint="policy",
    )
    assert result[date(2025, 4, 1)].nodes == 17
    assert result[date(2025, 4, 1)].edges == 30
    with pytest.raises(ValueError, match="different archive"):
        load_temporal_sample_sizes(
            tmp_path,
            dataset_fingerprint="wrong",
            sampling_policy_fingerprint="policy",
        )


def test_runner_uses_temporal_budget_sampler_and_single_pinning_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    days = [date(2025, 4, 1), date(2025, 4, 2)]
    rows = [
        {
            "day": day.isoformat(),
            "sample_nodes": {"player": 20, "team": 10, "game": 40},
            "sample_edges": {"team_game_event": 80},
            "sample_fingerprint": hashlib.sha256(day.isoformat().encode()).hexdigest(),
        }
        for day in days
    ]
    rows_json = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "graph_schema": "temporal_v7",
                "fingerprint": "dataset",
                "sampling_policy_fingerprint": "policy",
                "temporal_batching": {
                    "max_nodes_per_batch": 100,
                    "max_edges_per_batch": 200,
                    "max_days_per_batch": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "sample_index.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "sample_fingerprint_scope": "all_materialized_arrays_v2",
                "dataset_fingerprint": "dataset",
                "sampling_policy_fingerprint": "policy",
                "days": rows,
                "fingerprint": hashlib.sha256(rows_json.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    class _Generator:
        def manual_seed(self, _value: int) -> _Generator:
            return self

    def fake_loader(dataset: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(dataset=dataset, **kwargs)
        return captured

    fake_torch = SimpleNamespace(
        Generator=_Generator,
        utils=SimpleNamespace(data=SimpleNamespace(DataLoader=fake_loader)),
    )
    monkeypatch.setattr(kbo_runner, "require_torch", lambda: (fake_torch, None))
    config = kbo_runner.KBOTrainingConfig(
        device="cuda:0",
        workers=2,
        chronological=True,
        batch_days=8,
    )
    result = kbo_runner._loader(tmp_path, days, config, epoch=0, training=True)

    assert result["pin_memory"] is False
    assert result["persistent_workers"] is True
    assert result["prefetch_factor"] == 2
    assert list(result["batch_sampler"]) == [[0], [1]]
    assert "batch_size" not in result
