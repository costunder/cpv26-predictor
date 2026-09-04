from __future__ import annotations

import json
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pytest

import cpv26.data.kbo_temporal_archive as temporal_module
import cpv26.data.kbo_temporal_columnar as columnar_module
from cpv26.data.kbo_dataset_loader import open_kbo_graph_dataset
from cpv26.data.kbo_temporal_archive import (
    TEMPORAL_ROUTE_FEATURE_NAMES,
    KBOTemporalGraphDataset,
    TemporalSamplingPolicy,
    build_kbo_temporal_archive,
    build_kbo_temporal_sample_index,
)
from test_kbo_graph_dataset import _database, _game, _pa

_KST = ZoneInfo("Asia/Seoul")


def _route_input_copy(graph: object) -> dict[str, np.ndarray[Any, Any]]:
    arrays = graph.arrays  # type: ignore[attr-defined]
    return {
        name: value.copy()
        for name, value in arrays.items()
        if "__" in name or name in {"team_features", "game_features"}
    }


def test_temporal_archive_derives_event_graph_and_three_hop_history(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    archive = build_kbo_temporal_archive(database, tmp_path / "temporal")

    assert archive.manifest["dataset_version"] == 7
    assert archive.manifest["graph_schema"] == "temporal_v7"
    assert archive.manifest["materialization_contract_version"] == 6
    assert archive.manifest["fingerprint"] != archive.manifest["build_fingerprint"]
    assert archive.manifest["day_summary_contract"] == "trainer_split_summary_v1"
    assert archive.manifest["sampling_policy"] == {
        "history_scope": "all_prior_records",
        "recency_reference_days": 365,
    }
    assert archive.manifest["history_coverage"] == {
        "scope": "all_prior_records",
        "eligible_source_records": archive.manifest["record_count"],
        "stored_source_records": archive.manifest["record_count"],
        "stored_fraction": 1.0,
        "semantic_reduction": False,
    }
    assert archive.manifest["streaming_policy"] == {
        "record_storage": "archive_global_columnar_numpy_npy",
        "record_encoding": "pickle_free_utf8_blob_and_numeric_arrays",
        "read_mode": "read_only_mmap_shared_page_cache",
        "decode": "changed_records_only_without_persistent_source_dicts",
        "history_cursor": "cutoff_prefix_key_change_stream",
        "rewind": "replay_compact_key_changes_without_raw_history_decode",
        "materialized_graph_lifetime": "physical_batch_scoped",
        "raw_record_residency": "zero_decoded_source_records_per_worker",
        "worker_memory_sharing": True,
        "dense_graph_materialization": "once_per_sample_access",
        "per_day_full_history_cache": False,
        "production_ready": True,
    }
    assert archive.manifest["route_feature_dims"] == {
        name: len(features) for name, features in TEMPORAL_ROUTE_FEATURE_NAMES.items()
    }
    assert not (archive.directory / "days").exists()

    first = archive.load_day("2023-04-01")
    first_summary = next(
        entry for entry in archive.manifest["days"] if entry["day"] == "2023-04-01"
    )
    assert first_summary["games"] == len(first.match_targets)
    assert first_summary["live_hit_queries"] == len(first.live_hit_pa)
    assert first_summary["pa_queries"] == len(first.pa_targets)
    assert first_summary["box_pa_queries"] == len(first.box_pa_counts)
    assert first_summary["box_pa_outcomes"] == int(first.box_pa_counts.sum())
    assert first_summary["box_pitch_queries"] == len(first.box_pitch_targets)
    assert first_summary["box_pitch_observed_counts"] == int(first.box_pitch_mask.sum())
    current_first = set(first.match_game_index.tolist())
    assert len(first.routes["team_game_event"]["source_index"]) == 2
    assert not len(first.routes["batter_game_event"]["source_index"])
    assert not len(first.routes["pitcher_game_event"]["source_index"])
    assert not len(first.routes["batter_pa_pitcher_event"]["source_index"])

    second = archive.load_day("2023-04-02")
    current = set(second.match_game_index.tolist())
    assert current and current.isdisjoint(
        second.routes["batter_game_event"]["destination_index"].tolist()
    )
    assert current.isdisjoint(second.routes["pitcher_game_event"]["destination_index"].tolist())
    team_route = second.routes["team_game_event"]
    selected = np.isin(team_route["destination_index"], list(current))
    assert selected.sum() == 2
    assert team_route["event_features"][selected].tolist() == [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ]
    assert len(second.routes["batter_pa_pitcher_event"]["source_index"]) == 1
    assert second.routes["batter_pa_pitcher_event"]["event_features"].shape[1] == 17

    # Current game -> seed team -> historical game -> historical player.
    current_game = ("game", next(iter(current)))
    historical_game_index = second.game_ids.index("g1")
    old_batter_index = second.player_ids.index("old-batter")
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
    route_types = {
        "team_game_event": ("team", "game"),
        "batter_game_event": ("player", "game"),
        "pitcher_game_event": ("player", "game"),
        "batter_pa_pitcher_event": ("player", "player"),
    }
    for name, (source_type, destination_type) in route_types.items():
        route = second.routes[name]
        for source, destination in zip(
            route["source_index"], route["destination_index"], strict=True
        ):
            left = (source_type, int(source))
            right = (destination_type, int(destination))
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
    frontier = {current_game}
    visited = set(frontier)
    for _ in range(3):
        frontier = {
            neighbor
            for node in frontier
            for neighbor in adjacency.get(node, ())
            if neighbor not in visited
        }
        visited.update(frontier)
    assert ("game", historical_game_index) in visited
    assert ("player", old_batter_index) in visited
    assert current_first == {first.game_ids.index("g1")}


def test_temporal_archive_preserves_history_older_than_one_year_without_caps(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    with duckdb.connect(str(database)) as connection:
        _game(connection, "old-game", 1, season=2021)
        _pa(
            connection,
            "old-pa",
            "old-game",
            1,
            "historic-batter",
            "historic-pitcher",
            "double",
            season=2021,
        )
    archive = build_kbo_temporal_archive(database, tmp_path / "temporal-full-history")

    graph = archive.load_day("2023-04-03")
    current = {graph.game_ids[index] for index in graph.match_game_index.tolist()}
    historical = set(graph.game_ids) - current
    assert historical == {"old-game", "g1", "g2"}
    assert len(historical) == 3
    assert len(graph.routes["team_game_event"]["source_index"]) == 8
    assert len(graph.routes["batter_pa_pitcher_event"]["source_index"]) == 4
    assert "historic-batter" in graph.player_ids
    assert "historic-pitcher" in graph.player_ids
    assert archive.manifest["record_count"] >= 9
    for route_name, player_id in (
        ("batter_game_event", "historic-batter"),
        ("pitcher_game_event", "historic-pitcher"),
    ):
        route = graph.routes[route_name]
        endpoints = {
            (graph.player_ids[source], graph.game_ids[destination])
            for source, destination in zip(
                route["source_index"], route["destination_index"], strict=True
            )
        }
        assert (player_id, "old-game") in endpoints


def test_columnar_prefix_graphs_match_dense_history_reference_exactly(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database, include_pas=False)
    with duckdb.connect(str(database)) as connection:
        _game(connection, "old-game", 1, season=2021)
        _pa(
            connection,
            "old-pa",
            "old-game",
            1,
            "historic-batter",
            "historic-pitcher",
            "double",
            season=2021,
        )
        _pa(connection, "pa1", "g1", 1, "batter", "pitcher", "single")
        _pa(
            connection,
            "pa1",
            "g1",
            1,
            "batter",
            "pitcher",
            "strikeout",
            row_id="pa1-correction",
            available_at=datetime(2023, 4, 2, 12, tzinfo=_KST),
        )
        connection.execute(
            """
            UPDATE observed_plate_appearance
            SET valid_from=TIMESTAMPTZ '2023-04-02 12:00:00+09'
            WHERE observed_pa_row_id='pa1-correction'
            """
        )
        connection.execute(
            """
            UPDATE observed_plate_appearance
            SET valid_to=TIMESTAMPTZ '2023-04-02 12:00:00+09'
            WHERE observed_pa_row_id='pa1'
            """
        )
    snapshot = datetime(2025, 1, 1, tzinfo=_KST)
    archive = build_kbo_temporal_archive(
        database,
        tmp_path / "temporal-differential",
        knowledge_at=snapshot,
    )
    records, _, _ = temporal_module._read_records(database, snapshot)  # type: ignore[attr-defined]
    labels = temporal_module._label_records(records, snapshot)  # type: ignore[attr-defined]
    final_day = archive.days()[-1]
    history_span = (final_day - min(record.day for record in records)).days + 1
    reference_history = temporal_module._History(  # type: ignore[attr-defined]
        list(records),
        history_span,
        graph_schema="vnext",
        knowledge_cutoff_uses_ingested_at=True,
    )

    for day in archive.days():
        reference_history.advance(day)
        rows = labels[day]
        descriptors = tuple(
            temporal_module._QueryDescriptor(  # type: ignore[attr-defined]
                day,
                str(record.data["game_id"]),
                str(record.data["home_team_id"]),
                str(record.data["away_team_id"]),
            )
            for record in rows
            if record.kind == "game"
        )
        topology = temporal_module._select_topology(reference_history)  # type: ignore[attr-defined]
        reference = temporal_module._materialize_graph(  # type: ignore[attr-defined]
            day,
            reference_history,
            topology,
            descriptors,
            rows,
            archive.policy,
        )
        actual = archive.load_day(day)
        assert actual.player_ids == reference.player_ids
        assert actual.team_ids == reference.team_ids
        assert actual.game_ids == reference.game_ids
        assert actual.arrays.keys() == reference.arrays.keys()
        for name in actual.arrays:
            np.testing.assert_array_equal(actual.arrays[name], reference.arrays[name])


def test_sample_materialization_has_no_per_day_or_raw_history_cache(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    archive = build_kbo_temporal_archive(database, tmp_path / "temporal-bounded")

    for day in archive.days():
        graph = archive.load_day(day)
        assert graph.day == day

    history = archive._stream_history  # type: ignore[attr-defined]
    assert history is not None
    assert history.raw_python_record_residency == 0
    assert history._active_record_by_key.shape == (  # type: ignore[attr-defined]
        archive._record_store.key_count,  # type: ignore[attr-defined]
    )
    assert history._prefix_position <= len(  # type: ignore[attr-defined]
        archive._record_store.arrays["schedule_day_ordinal"]  # type: ignore[attr-defined]
    )
    assert not hasattr(archive, "_graphs_cache")
    assert not hasattr(archive, "_stream_records")
    assert not (archive.directory / "days").exists()


def test_dataset_worker_pickle_reopens_mmaps_without_serializing_record_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    archive = build_kbo_temporal_archive(database, tmp_path / "temporal-worker")
    archive.load_day("2023-04-02")

    payload = pickle.dumps(archive)
    state = archive.__getstate__()
    assert "_record_store" not in state
    assert state["_record_store_attestation"]
    assert state["_stream_history"] is None
    assert len(payload) < 100_000

    def unexpected_full_checksum(_path: Path) -> str:
        raise AssertionError("spawned workers must reuse the parent file attestation")

    monkeypatch.setattr(columnar_module, "_sha256_file", unexpected_full_checksum)
    restored = pickle.loads(payload)
    assert restored._record_store.mmap_backed  # type: ignore[attr-defined]
    assert restored._stream_history is None  # type: ignore[attr-defined]
    graph = restored.load_day("2023-04-02")
    assert graph.day_id == "2023-04-02"
    assert restored._stream_history.raw_python_record_residency == 0  # type: ignore[union-attr]


def test_current_results_and_participants_cannot_select_topology(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    before = build_kbo_temporal_archive(database, tmp_path / "before").load_day("2023-04-02")
    with duckdb.connect(str(database)) as connection:
        connection.execute("UPDATE game SET home_score=99 WHERE game_id='g2'")
        connection.execute(
            """
            UPDATE observed_plate_appearance
            SET outcome='strikeout', is_at_bat=true, is_hit=false, total_bases=0
            WHERE plate_appearance_id='pa2'
            """
        )
        _pa(
            connection,
            "same-day-extra",
            "g2",
            2,
            "current-only-batter",
            "current-only-pitcher",
            "double",
        )
    after = build_kbo_temporal_archive(database, tmp_path / "after").load_day("2023-04-02")

    before_inputs = _route_input_copy(before)
    after_inputs = _route_input_copy(after)
    assert before_inputs.keys() == after_inputs.keys()
    for name in before_inputs:
        np.testing.assert_array_equal(before_inputs[name], after_inputs[name])
    assert before.match_runs.tolist() != after.match_runs.tolist()
    assert before.pa_targets.tolist() != after.pa_targets.tolist()
    player_endpoint_columns = {
        "batter_game_event": ("source_index",),
        "pitcher_game_event": ("source_index",),
        "batter_pa_pitcher_event": ("source_index", "destination_index"),
    }
    for player in ("current-only-batter", "current-only-pitcher"):
        index = after.player_ids.index(player)
        assert all(
            index not in after.routes[route][column].tolist()
            for route, columns in player_endpoint_columns.items()
            for column in columns
        )


def test_publication_and_revision_cutoffs_select_one_raw_pa_version(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database, include_pas=False)
    with duckdb.connect(str(database)) as connection:
        _pa(connection, "pa1", "g1", 1, "batter", "pitcher", "single")
        _pa(
            connection,
            "pa1",
            "g1",
            1,
            "batter",
            "pitcher",
            "strikeout",
            row_id="pa1-correction",
            available_at=datetime(2023, 4, 2, 12, tzinfo=_KST),
        )
        connection.execute(
            """
            UPDATE observed_plate_appearance
            SET valid_from=TIMESTAMPTZ '2023-04-02 12:00:00+09'
            WHERE observed_pa_row_id='pa1-correction'
            """
        )
        connection.execute(
            """
            UPDATE observed_plate_appearance
            SET valid_to=TIMESTAMPTZ '2023-04-02 12:00:00+09'
            WHERE observed_pa_row_id='pa1'
            """
        )
    archive = build_kbo_temporal_archive(database, tmp_path / "temporal")
    second = archive.load_day("2023-04-02")
    third = archive.load_day("2023-04-03")
    second_event = second.routes["batter_pa_pitcher_event"]["event_features"]
    third_event = third.routes["batter_pa_pitcher_event"]["event_features"]
    assert second_event.shape == third_event.shape == (1, 17)
    assert second_event[0, 2] == 1  # original single is still valid at midnight
    assert second_event[0, 5] == 0
    assert third_event[0, 2] == 0
    assert third_event[0, 5] == 1  # noon correction enters only the next cutoff


def test_late_ingestion_cannot_leak_through_backdated_availability(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database, include_pas=False)
    with duckdb.connect(str(database)) as connection:
        _pa(connection, "pa1", "g1", 1, "batter", "pitcher", "single")
        connection.execute(
            """
            UPDATE observed_plate_appearance
            SET ingested_at=TIMESTAMPTZ '2023-04-02 12:00:00+09'
            WHERE observed_pa_row_id='pa1'
            """
        )

    archive = build_kbo_temporal_archive(database, tmp_path / "temporal")
    before_ingestion = archive.load_day("2023-04-02")
    after_ingestion = archive.load_day("2023-04-03")

    assert not len(before_ingestion.routes["batter_pa_pitcher_event"]["source_index"])
    assert len(after_ingestion.routes["batter_pa_pitcher_event"]["source_index"]) == 1


def test_archive_columns_are_unique_pickle_free_mmaps_and_labels_are_indices(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    archive = build_kbo_temporal_archive(database, tmp_path / "temporal")

    store = archive._record_store  # type: ignore[attr-defined]
    assert store.mmap_backed
    assert store.record_count == archive.manifest["record_count"]
    assert store.allowed_record_count == store.record_count
    record_keys = [store.record_key(index) for index in range(store.record_count)]
    assert len(record_keys) == len(set(record_keys))
    for array in store.arrays.values():
        assert isinstance(array, np.memmap)
        assert not array.dtype.hasobject
        assert array.flags.writeable is False
    for column in store.strings.values():
        assert isinstance(column.blob, np.memmap)
        assert isinstance(column.offsets, np.memmap)
        assert column.blob.flags.writeable is False
        assert column.offsets.flags.writeable is False
    for entry in archive.manifest["label_shards"]:
        with np.load(archive.directory / entry["file"], allow_pickle=False) as shard:
            assert set(shard.files) == {"day", "record_index"}
            assert shard["record_index"].dtype == np.int64
    for entry in archive.manifest["query_shards"]:
        with np.load(archive.directory / entry["file"], allow_pickle=False) as shard:
            assert set(shard.files) == {
                "day",
                "game_id",
                "home_team_id",
                "away_team_id",
            }


def test_temporal_archive_reuses_identical_logical_build(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    output = tmp_path / "temporal"
    first = build_kbo_temporal_archive(database, output)

    second = build_kbo_temporal_archive(database, output)

    assert second.directory == first.directory
    assert second.manifest["build_fingerprint"] == first.manifest["build_fingerprint"]
    assert second.manifest["fingerprint"] == first.manifest["fingerprint"]


def test_temporal_archive_artifact_fingerprint_binds_shard_entries(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    archive = build_kbo_temporal_archive(database, tmp_path / "temporal")
    manifest_path = archive.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_store"]["columns"]["values"]["sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact fingerprint is inconsistent"):
        KBOTemporalGraphDataset(archive.directory)


@pytest.mark.parametrize("field", ["fingerprint", "build_fingerprint"])
def test_temporal_archive_requires_lowercase_manifest_fingerprints(
    tmp_path: Path, field: str
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    archive = build_kbo_temporal_archive(database, tmp_path / "temporal")
    manifest_path = archive.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = "A" + str(manifest[field])[1:]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        KBOTemporalGraphDataset(archive.directory)


def test_sample_index_and_label_ceiling_never_open_held_out_year(
    tmp_path: Path,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    with duckdb.connect(str(database)) as connection:
        _game(connection, "held-out", 1, season=2024)
    archive = build_kbo_temporal_archive(database, tmp_path / "temporal")
    sealed = KBOTemporalGraphDataset(archive.directory, label_year_ceiling=2023)
    assert all(day.year == 2023 for day in sealed.days())
    store = sealed._record_store  # type: ignore[attr-defined]
    assert store.allowed_record_count < store.record_count
    sealed.load_day("2023-04-02")
    future_index = next(
        index
        for index, value in enumerate(store.arrays["day_ordinal"])
        if date.fromordinal(int(value)).year == 2024
    )
    with pytest.raises(PermissionError, match="sealed"):
        store.ref(future_index)
    with pytest.raises(PermissionError, match="sealed"):
        sealed.load_day("2024-04-01")
    with pytest.raises(ValueError, match="end_day or label_year_ceiling"):
        build_kbo_temporal_sample_index(archive)

    report = build_kbo_temporal_sample_index(
        archive,
        label_year_ceiling=2023,
    )
    assert report["label_year_ceiling"] == 2023
    assert report["held_out_labels_loaded"] is False
    assert report["date_end"] == "2023-04-03"
    assert {entry["day"] for entry in report["days"]} == {
        "2023-04-01",
        "2023-04-02",
        "2023-04-03",
    }
    assert not any(entry["day"].startswith("2024-") for entry in report["days"])
    assert (archive.directory / "sample_index.json").is_file()


def test_full_history_is_deterministic_and_reuses_one_mmap_prefix_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    with duckdb.connect(str(database)) as connection:
        _game(connection, "next-season", 1, season=2024)
    archive = build_kbo_temporal_archive(database, tmp_path / "temporal")
    original: Any = temporal_module._MMapHistory  # type: ignore[attr-defined]
    constructions = 0

    class CountingHistory(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            nonlocal constructions
            constructions += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(temporal_module, "_MMapHistory", CountingHistory)
    dataset = KBOTemporalGraphDataset(archive.directory)
    first = dataset.load_day(date(2023, 4, 1))
    second = dataset.load_day(date(2023, 4, 2))
    third = dataset.load_day(date(2023, 4, 3))
    next_season = dataset.load_day(date(2024, 4, 1))
    assert constructions == 1
    repeat = dataset.load_day(date(2023, 4, 2))
    assert constructions == 1  # rewind resets integer state; it never remaps or decodes history
    assert dataset._stream_history.raw_python_record_residency == 0  # type: ignore[union-attr]
    assert not hasattr(dataset, "_stream_records")
    assert not hasattr(dataset, "_records_cache")
    for left, right in ((second, repeat),):
        assert left.player_ids == right.player_ids
        assert left.team_ids == right.team_ids
        assert left.game_ids == right.game_ids
        for name in left.arrays:
            np.testing.assert_array_equal(left.arrays[name], right.arrays[name])
    assert first.day_id == "2023-04-01"
    assert third.day_id == "2023-04-03"
    assert next_season.day_id == "2024-04-01"


@pytest.mark.parametrize(
    "options",
    [
        {"recency_reference_days": 0},
        {"recency_reference_days": True},
    ],
)
def test_temporal_sampling_policy_is_fail_closed(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TemporalSamplingPolicy(**options)


def test_temporal_sampling_policy_rejects_any_partial_history_scope() -> None:
    with pytest.raises(ValueError, match="all_prior_records"):
        TemporalSamplingPolicy(history_scope="bounded")


def test_archive_rejects_legacy_bounded_history_contract(tmp_path: Path) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database)
    archive = build_kbo_temporal_archive(database, tmp_path / "temporal")
    manifest_path = archive.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sampling_policy"] = {
        "lookback_days": 365,
        "max_games_per_seed_team": 160,
        "max_games_per_player": 48,
        "max_historical_games_total": 160,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lossless full-history policy"):
        KBOTemporalGraphDataset(archive.directory)


def test_validation_ceiling_does_not_decode_2026_raw_record_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "canonical.duckdb"
    _database(database, season=2025)
    with duckdb.connect(str(database)) as connection:
        _game(connection, "held-out-2026", 1, season=2026)
    archive = build_kbo_temporal_archive(database, tmp_path / "temporal")
    original_mapping: Any = columnar_module._RecordDataView._mapping  # type: ignore[attr-defined]
    decoded_years: list[int] = []

    def counting_mapping(view: Any) -> dict[str, Any]:
        year = date.fromordinal(
            int(view._store.arrays["day_ordinal"][view._index])
        ).year
        decoded_years.append(year)
        return original_mapping(view)

    monkeypatch.setattr(columnar_module._RecordDataView, "_mapping", counting_mapping)
    sealed = open_kbo_graph_dataset(
        archive.directory,
        label_year_ceiling=2025,
    )
    sealed.load_day("2025-04-03")

    assert decoded_years
    assert max(decoded_years) <= 2025
    with pytest.raises(PermissionError, match="sealed"):
        sealed.load_day("2026-04-01")
