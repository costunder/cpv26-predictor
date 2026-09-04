from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cpv26.models.kbo_relgnn import (  # noqa: E402
    KBO_ROUTE_NAMES,
    KBO_TEMPORAL_ROUTE_NAMES,
    KBO_VNEXT_ROUTE_NAMES,
    KBORelGNNConfig,
    KBORelGNNModel,
    collate_kbo_day_graphs,
    kbo_route_registry,
)


def _route(
    rng: np.random.Generator,
    source: list[int],
    destination: list[int],
    width: int,
    *,
    current: tuple[bool, ...] | None = None,
) -> dict[str, np.ndarray[Any, Any]]:
    count = len(source)
    current_mask = np.asarray(current or (False,) * count, dtype=bool)
    ages = np.where(current_mask, 0, 86_400).astype(np.float32)
    delays = np.where(current_mask, 0, 3_600).astype(np.float32)
    return {
        "source_index": np.asarray(source, dtype=np.int64),
        "destination_index": np.asarray(destination, dtype=np.int64),
        "event_features": rng.normal(size=(count, width)).astype(np.float32),
        "event_age_seconds": ages,
        "publication_delay_seconds": delays,
        "weights": np.ones(count, dtype=np.float32),
    }


def _temporal_day(day_id: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    day: dict[str, Any] = {
        "day_id": day_id,
        "node_features": {
            "player": rng.normal(size=(3, 4)).astype(np.float32),
            "team": rng.normal(size=(2, 8)).astype(np.float32),
            "game": rng.normal(size=(3, 9)).astype(np.float32),
        },
        "role_features": {
            "batting": rng.normal(size=(3, 8)).astype(np.float32),
            "pitching": rng.normal(size=(3, 8)).astype(np.float32),
        },
        "routes": {
            "batter_game_event": _route(rng, [0, 2], [0, 1], 6),
            "pitcher_game_event": _route(rng, [1, 1], [0, 1], 6),
            "team_game_event": _route(
                rng,
                [0, 1, 0, 1],
                [0, 0, 2, 2],
                4,
                current=(False, False, True, True),
            ),
            "batter_pa_pitcher_event": _route(rng, [0, 2], [1, 1], 17),
        },
        "match_home_team_index": np.asarray([0], dtype=np.int64),
        "match_away_team_index": np.asarray([1], dtype=np.int64),
        "match_game_index": np.asarray([2], dtype=np.int64),
        "match_targets": np.asarray([2], dtype=np.int64),
        "match_runs": np.asarray([[4, 2]], dtype=np.float32),
        "match_query_ids": (f"{day_id}:game",),
        "live_hit_player_index": np.asarray([0], dtype=np.int64),
        "live_hit_team_index": np.asarray([0], dtype=np.int64),
        "live_hit_opponent_index": np.asarray([1], dtype=np.int64),
        "live_hit_game_index": np.asarray([2], dtype=np.int64),
        "live_hit_pa": np.asarray([4], dtype=np.int64),
        "live_hit_pa_min": np.asarray([4], dtype=np.int64),
        "live_hit_hits": np.asarray([1], dtype=np.int64),
        "live_hit_query_ids": (f"{day_id}:live",),
        "pa_batter_index": np.asarray([0], dtype=np.int64),
        "pa_pitcher_index": np.asarray([1], dtype=np.int64),
        "pa_game_index": np.asarray([2], dtype=np.int64),
        "pa_targets": np.asarray([2], dtype=np.int64),
        "pa_context": rng.normal(size=(1, 10)).astype(np.float32),
        "pa_query_ids": (f"{day_id}:pa",),
        "box_pa_player_index": np.empty(0, dtype=np.int64),
        "box_pa_team_index": np.empty(0, dtype=np.int64),
        "box_pa_opponent_index": np.empty(0, dtype=np.int64),
        "box_pa_game_index": np.empty(0, dtype=np.int64),
        "box_pa_counts": np.empty((0, 10), dtype=np.float32),
        "box_pa_query_ids": (),
        "box_pitch_player_index": np.empty(0, dtype=np.int64),
        "box_pitch_team_index": np.empty(0, dtype=np.int64),
        "box_pitch_opponent_index": np.empty(0, dtype=np.int64),
        "box_pitch_game_index": np.empty(0, dtype=np.int64),
        "box_pitch_targets": np.empty((0, 10), dtype=np.float32),
        "box_pitch_mask": np.empty((0, 10), dtype=bool),
        "box_pitch_query_ids": (),
    }
    return day


def _temporal_config() -> KBORelGNNConfig:
    return KBORelGNNConfig(
        node_feature_dims={"player": 4, "team": 8, "game": 9},
        role_feature_dims={"batting": 8, "pitching": 8},
        route_feature_dims={
            "batter_game_event": 6,
            "pitcher_game_event": 6,
            "team_game_event": 4,
            "batter_pa_pitcher_event": 17,
        },
        hidden_dim=16,
        num_layers=3,
        num_attention_heads=4,
        dropout=0.0,
        compact_kbo_channels=True,
    )


def test_temporal_v7_registry_is_exact_and_role_aware() -> None:
    assert KBO_TEMPORAL_ROUTE_NAMES == (
        "batter_game_event",
        "pitcher_game_event",
        "team_game_event",
        "batter_pa_pitcher_event",
    )
    registry = kbo_route_registry()
    expected = {
        "batter_game_event": ("player", "batting", "game", "shared"),
        "pitcher_game_event": ("player", "pitching", "game", "shared"),
        "team_game_event": ("team", "shared", "game", "shared"),
        "batter_pa_pitcher_event": ("player", "batting", "player", "pitching"),
    }
    for name, endpoints in expected.items():
        route = registry.require(name)
        assert (
            route.source_type,
            route.source_role,
            route.destination_type,
            route.destination_role,
        ) == endpoints


def test_temporal_v7_collates_offsets_and_uses_current_query_game_context() -> None:
    days = [_temporal_day("2024-04-01", 1), _temporal_day("2024-04-02", 2)]
    batch = collate_kbo_day_graphs(days)
    assert batch["match_game_index"].tolist() == [2, 5]
    assert batch["live_hit_game_index"].tolist() == [2, 5]
    assert batch["pa_game_index"].tolist() == [2, 5]
    assert batch["node_graph_index"]["game"].tolist() == [0, 0, 0, 1, 1, 1]
    team_games = next(
        route for route in batch["routes"] if route.route_name == "team_game_event"
    )
    assert team_games.source_index.tolist() == [0, 1, 0, 1, 2, 3, 2, 3]
    assert team_games.destination_index.tolist() == [0, 0, 2, 2, 3, 3, 5, 5]

    torch.manual_seed(19)
    model: Any = KBORelGNNModel(_temporal_config())
    model.eval()
    with torch.no_grad():
        original = model({**batch, "routes": ()})["match_logits"]
        changed_batch = copy.deepcopy(batch)
        changed_batch["node_features"]["game"][batch["match_game_index"]] += 3.0
        changed = model({**changed_batch, "routes": ()})["match_logits"]
    assert torch.isfinite(model(batch)["match_logits"]).all()
    assert not torch.allclose(original, changed)


def test_temporal_v7_requires_its_complete_unmixed_route_family() -> None:
    dimensions = dict(_temporal_config().route_feature_dims)
    missing = dict(dimensions)
    missing.pop("batter_pa_pitcher_event")
    with pytest.raises(ValueError, match="exact reviewed route set"):
        KBORelGNNConfig(
            node_feature_dims={"player": 4, "team": 8, "game": 9},
            role_feature_dims={"batting": 8, "pitching": 8},
            route_feature_dims=missing,
        )

    mixed = {**dimensions, "team_game_context": 4}
    with pytest.raises(ValueError, match="exact reviewed route set"):
        KBORelGNNConfig(
            node_feature_dims={"player": 4, "team": 8, "game": 9},
            role_feature_dims={"batting": 8, "pitching": 8},
            route_feature_dims=mixed,
        )

    malformed_day = _temporal_day("2024-04-01", 3)
    malformed_day["routes"].pop("pitcher_game_event")
    with pytest.raises(ValueError, match="exact reviewed route set"):
        collate_kbo_day_graphs([malformed_day])


def test_temporal_v7_rejects_legacy_dense_inactive_channels() -> None:
    with pytest.raises(ValueError, match="compact_kbo_channels=True"):
        replace(_temporal_config(), compact_kbo_channels=False)


def test_temporal_extension_preserves_legacy_config_fingerprints() -> None:
    configurations = (
        KBORelGNNConfig(
            node_feature_dims={"player": 4, "team": 8},
            role_feature_dims={"batting": 8, "pitching": 8},
            route_feature_dims=dict.fromkeys(KBO_ROUTE_NAMES, 6),
            hidden_dim=16,
            num_layers=2,
            num_attention_heads=4,
            dropout=0.0,
            include_boxscore_heads=True,
        ),
        KBORelGNNConfig(
            node_feature_dims={"player": 4, "team": 8, "game": 4},
            role_feature_dims={"batting": 8, "pitching": 8},
            route_feature_dims={
                name: 4 if name == "team_game_context" else 6
                for name in KBO_VNEXT_ROUTE_NAMES
            },
            hidden_dim=16,
            num_layers=2,
            num_attention_heads=4,
            dropout=0.0,
            include_boxscore_heads=True,
        ),
    )
    expected = (
        "44f8b970e40bda562bbf51b9b8525e9bc441dd99a1ddf6dc21f448d5d69ba1fe",
        "5bdd58b17743ec37d3a9d56605d92b0411a1f696fb073e66534e51e9f730456f",
    )
    observed = tuple(
        hashlib.sha256(
            json.dumps(
                config.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        for config in configurations
    )
    assert observed == expected
