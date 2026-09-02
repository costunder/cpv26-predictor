from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cpv26.data.kbo_graph_audit import (
    analyze_kbo_graph_dataset,
    audit_kbo_graph_dataset,
)
from cpv26.data.kbo_graph_dataset import ROUTE_METADATA, KBOGraphDataset


def _route_features(route: str, count: int) -> np.ndarray[Any, Any]:
    scale = 20 if route == "home_team_game_away_team" else 100
    encoded = math.log1p(count) / math.log1p(scale)
    return np.asarray([[encoded, 0, 0, 0, 0, 0.5]], dtype=np.float32)


def _make_dataset(
    directory: Path,
    *,
    version: int = 5,
    history_rows: int | None = 10,
    zero_count_route: str | None = None,
) -> Path:
    day = "2023-04-02"
    player_ids = np.asarray(["p0", "p1", "p2"], dtype=np.str_)
    team_ids = np.asarray(["t0", "t1"], dtype=np.str_)
    arrays: dict[str, Any] = {
        "_player_ids": player_ids,
        "_team_ids": team_ids,
        "player_features": np.zeros((3, 4), dtype=np.float32),
        "player_batting_features": np.zeros((3, 8), dtype=np.float32),
        "player_pitching_features": np.zeros((3, 8), dtype=np.float32),
        "team_features": np.zeros((2, 8), dtype=np.float32),
        "match_home_team_index": np.asarray([0], dtype=np.int64),
        "match_away_team_index": np.asarray([1], dtype=np.int64),
        "match_targets": np.asarray([2], dtype=np.int64),
        "match_runs": np.asarray([[4, 2]], dtype=np.float32),
        "match_query_ids": np.asarray(["g0"], dtype=np.str_),
        # The queried player is intentionally isolated; both team endpoints are connected.
        "live_hit_player_index": np.asarray([2], dtype=np.int64),
        "live_hit_team_index": np.asarray([0], dtype=np.int64),
        "live_hit_opponent_index": np.asarray([1], dtype=np.int64),
        "live_hit_pa": np.asarray([3], dtype=np.int64),
        "live_hit_hits": np.asarray([1], dtype=np.int64),
        "live_hit_query_ids": np.asarray(["g0|p2"], dtype=np.str_),
        "pa_batter_index": np.asarray([0], dtype=np.int64),
        "pa_pitcher_index": np.asarray([1], dtype=np.int64),
        "pa_targets": np.asarray([2], dtype=np.int64),
        "pa_context": np.zeros((1, 10), dtype=np.float32),
        "pa_query_ids": np.asarray(["pa0"], dtype=np.str_),
    }
    endpoints = {
        "batter_pa_pitcher": (0, 1, 3),
        "batter_participation_team": (0, 0, 2),
        "pitcher_participation_team": (1, 1, 1),
        "home_team_game_away_team": (0, 1, 4),
    }
    for route, (source, destination, count) in endpoints.items():
        prefix = route + "__"
        arrays[prefix + "source_index"] = np.asarray([source], dtype=np.int64)
        arrays[prefix + "destination_index"] = np.asarray([destination], dtype=np.int64)
        arrays[prefix + "event_features"] = _route_features(
            route, 0 if zero_count_route == route else count
        )
        arrays[prefix + "event_age_seconds"] = np.asarray([86_400], dtype=np.float32)
        arrays[prefix + "publication_delay_seconds"] = np.asarray([0], dtype=np.float32)
        arrays[prefix + "weights"] = np.asarray([1], dtype=np.float32)

    days = directory / "days"
    days.mkdir(parents=True)
    archive = days / f"{day}.npz"
    np.savez_compressed(archive, **arrays)
    entry: dict[str, Any] = {
        "day": day,
        "file": f"days/{day}.npz",
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    if history_rows is not None:
        entry["history_rows"] = history_rows
    manifest = {
        "dataset_version": version,
        "days": [entry],
        "route_metadata": ROUTE_METADATA,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


@pytest.mark.parametrize("version", [2, 3, 4, 5])
def test_audit_supports_directory_and_dataset_for_v2_through_v5(
    tmp_path: Path, version: int
) -> None:
    directory = _make_dataset(tmp_path / f"graph-v{version}", version=version)
    before = {
        path.relative_to(directory).as_posix(): (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in directory.rglob("*")
        if path.is_file()
    }

    report = audit_kbo_graph_dataset(directory, end_day="2023-12-31")
    from_object = analyze_kbo_graph_dataset(
        KBOGraphDataset(directory), end_day="2023-12-31"
    )

    assert report == from_object
    assert report["dataset_version"] == version
    assert report["days"] == 1
    assert report["date_start"] == report["date_end"] == "2023-04-02"
    assert report["totals"]["node_occurrences"] == {"player": 3, "team": 2}
    assert report["by_year"]["2023"] == report["years"][0]
    assert report["years"][0]["year"] == 2023
    json.dumps(report, allow_nan=False)

    after = {
        path.relative_to(directory).as_posix(): (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in directory.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_audit_reports_edge_compression_and_query_reachability(tmp_path: Path) -> None:
    report = audit_kbo_graph_dataset(
        _make_dataset(tmp_path / "graph", history_rows=2), end_day="2023-12-31"
    )["totals"]

    compression = report["history_compression"]
    assert compression["raw_history_row_occurrences"] == 2
    assert compression["raw_relation_event_occurrences"] == 10
    assert compression["unique_edge_occurrences"] == 4
    assert compression["raw_rows_per_unique_edge"] == pytest.approx(2.5)
    assert compression["unique_edges_per_raw_row"] == pytest.approx(0.4)
    assert compression["compression_fraction"] == pytest.approx(0.6)

    batter_pitcher = report["routes"]["batter_pa_pitcher"]
    assert batter_pitcher["edges"] == 1
    assert batter_pitcher["nodes"] == 2
    assert batter_pitcher["unique_endpoint_pairs"] == 1
    assert batter_pitcher["encoded_raw_event_occurrences"] == 3
    assert batter_pitcher["history_compression"] == {
        "raw_row_occurrences": 3,
        "unique_edge_occurrences": 1,
        "raw_rows_per_unique_edge": pytest.approx(3.0),
        "unique_edges_per_raw_row": pytest.approx(1 / 3),
        "compression_fraction": pytest.approx(2 / 3),
    }

    queries = report["all_queries"]
    assert queries["query_node_occurrences"] == 7
    assert queries["unique_query_nodes"] == 5
    assert queries["isolated_query_node_occurrences"] == 1
    assert queries["isolation_fraction"] == pytest.approx(1 / 7)
    assert queries["degree"]["min"] == 0
    assert queries["degree"]["max"] == 2
    assert queries["degree"]["mean"] == pytest.approx(12 / 7)
    assert queries["one_hop_coverage"] == pytest.approx(3 / 7)
    assert queries["two_hop_coverage"] == pytest.approx(4.5 / 7)
    assert report["queries"]["live_hit"]["by_role"]["player"][
        "isolation_fraction"
    ] == 1.0


def test_missing_or_unrecoverable_raw_counts_are_explicitly_unknown(tmp_path: Path) -> None:
    directory = _make_dataset(
        tmp_path / "graph",
        history_rows=None,
        zero_count_route="batter_participation_team",
    )
    totals = audit_kbo_graph_dataset(directory, end_day="2023-12-31")["totals"]

    overall = totals["history_compression"]
    assert overall["raw_history_row_occurrences"] is None
    assert overall["history_rows_missing_days"] == 1
    assert overall["compression_fraction"] is None
    route = totals["routes"]["batter_participation_team"]
    assert route["encoded_raw_event_occurrences"] is None
    assert route["encoded_raw_event_occurrences_lower_bound"] == 0
    assert route["zero_encoded_count_edges"] == 1
    assert route["history_compression"]["compression_fraction"] is None


def test_audit_rejects_non_dataset_input() -> None:
    with pytest.raises(TypeError, match="KBOGraphDataset"):
        audit_kbo_graph_dataset(object())  # type: ignore[arg-type]


def test_audit_requires_an_explicit_held_out_safe_end_date(tmp_path: Path) -> None:
    directory = _make_dataset(tmp_path / "graph")
    with pytest.raises(ValueError, match="end_day is required"):
        audit_kbo_graph_dataset(directory)
