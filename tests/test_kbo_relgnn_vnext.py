from __future__ import annotations

import copy
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
    kbo_multitask_loss,
)


def _route(
    rng: np.random.Generator,
    source: list[int],
    destination: list[int],
    width: int,
    *,
    current: bool = False,
) -> dict[str, np.ndarray[Any, Any]]:
    count = len(source)
    return {
        "source_index": np.asarray(source, dtype=np.int64),
        "destination_index": np.asarray(destination, dtype=np.int64),
        "event_features": rng.normal(size=(count, width)).astype(np.float32),
        "event_age_seconds": np.full(count, 0 if current else 86_400, dtype=np.float32),
        "publication_delay_seconds": np.full(
            count, 0 if current else 3_600, dtype=np.float32
        ),
        "weights": np.ones(count, dtype=np.float32),
    }


def _vnext_day(day_id: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    routes = {
        "batter_pa_pitcher": _route(rng, [0], [1], 6),
        "batter_participation_team": _route(rng, [0], [0], 6),
        "pitcher_participation_team": _route(rng, [1], [1], 6),
        "home_team_game_away_team": _route(rng, [0], [1], 6),
        "batter_game_participation": _route(rng, [0], [0], 6),
        "pitcher_game_participation": _route(rng, [1], [0], 6),
        "team_game_context": _route(rng, [0, 1], [1, 1], 4, current=True),
    }
    day: dict[str, Any] = {
        "day_id": day_id,
        "node_features": {
            "player": rng.normal(size=(2, 4)).astype(np.float32),
            "team": rng.normal(size=(2, 8)).astype(np.float32),
            "game": rng.normal(size=(2, 4)).astype(np.float32),
        },
        "role_features": {
            "batting": rng.normal(size=(2, 8)).astype(np.float32),
            "pitching": rng.normal(size=(2, 8)).astype(np.float32),
        },
        "routes": routes,
        "match_home_team_index": np.asarray([0], dtype=np.int64),
        "match_away_team_index": np.asarray([1], dtype=np.int64),
        "match_game_index": np.asarray([1], dtype=np.int64),
        "match_targets": np.asarray([2], dtype=np.int64),
        "match_runs": np.asarray([[4, 2]], dtype=np.float32),
        "match_query_ids": (f"{day_id}:game",),
        "live_hit_player_index": np.asarray([0], dtype=np.int64),
        "live_hit_team_index": np.asarray([0], dtype=np.int64),
        "live_hit_opponent_index": np.asarray([1], dtype=np.int64),
        "live_hit_game_index": np.asarray([1], dtype=np.int64),
        "live_hit_pa": np.asarray([4], dtype=np.int64),
        "live_hit_pa_min": np.asarray([4], dtype=np.int64),
        "live_hit_hits": np.asarray([1], dtype=np.int64),
        "live_hit_query_ids": (f"{day_id}:live",),
        "pa_batter_index": np.asarray([0], dtype=np.int64),
        "pa_pitcher_index": np.asarray([1], dtype=np.int64),
        "pa_game_index": np.asarray([1], dtype=np.int64),
        "pa_targets": np.asarray([2], dtype=np.int64),
        "pa_context": rng.normal(size=(1, 10)).astype(np.float32),
        "pa_query_ids": (f"{day_id}:pa",),
        "box_pa_player_index": np.asarray([0], dtype=np.int64),
        "box_pa_team_index": np.asarray([0], dtype=np.int64),
        "box_pa_opponent_index": np.asarray([1], dtype=np.int64),
        "box_pa_game_index": np.asarray([1], dtype=np.int64),
        "box_pa_counts": np.asarray([[2, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
        "box_pa_query_ids": (f"{day_id}:box-pa",),
        "box_pitch_player_index": np.asarray([1], dtype=np.int64),
        "box_pitch_team_index": np.asarray([1], dtype=np.int64),
        "box_pitch_opponent_index": np.asarray([0], dtype=np.int64),
        "box_pitch_game_index": np.asarray([1], dtype=np.int64),
        "box_pitch_targets": np.asarray(
            [[5, 3, 21, 4, 1, 0, 1, 2, 1, 1]], dtype=np.float32
        ),
        "box_pitch_mask": np.ones((1, 10), dtype=bool),
        "box_pitch_query_ids": (f"{day_id}:box-pitch",),
    }
    for kind, count in (("player", 2), ("team", 2)):
        for role, width in (("batting", 19), ("pitching", 21)):
            day[f"{kind}_box_{role}_features"] = rng.random((count, width)).astype(
                np.float32
            )
    return day


def _config(*, vnext: bool) -> KBORelGNNConfig:
    route_widths = {
        name: 4 if name == "team_game_context" else 6
        for name in (KBO_VNEXT_ROUTE_NAMES if vnext else KBO_ROUTE_NAMES)
    }
    return KBORelGNNConfig(
        node_feature_dims={
            "player": 4,
            "team": 8,
            **({"game": 4} if vnext else {}),
        },
        role_feature_dims={"batting": 8, "pitching": 8},
        route_feature_dims=route_widths,
        hidden_dim=16,
        num_layers=2,
        num_attention_heads=4,
        dropout=0.0,
        include_boxscore_heads=True,
    )


def _legacy_day() -> dict[str, Any]:
    day = copy.deepcopy(_vnext_day("2024-04-01", 1))
    day["node_features"].pop("game")
    day["routes"] = {name: day["routes"][name] for name in KBO_ROUTE_NAMES}
    for task in ("match", "live_hit", "pa", "box_pa", "box_pitch"):
        day.pop(f"{task}_game_index")
    return day


def _parameter_count(model: KBORelGNNModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _temporal_day(seed: int = 19) -> dict[str, Any]:
    day = _vnext_day("2024-04-03", seed)
    rng = np.random.default_rng(seed + 1)
    day["routes"] = {
        "batter_game_event": _route(rng, [0, 0], [0, 1], 6),
        "pitcher_game_event": _route(rng, [1, 1], [0, 1], 6),
        "team_game_event": _route(rng, [0, 1], [0, 1], 6),
        "batter_pa_pitcher_event": _route(rng, [0, 0], [1, 1], 6),
    }
    return day


def _temporal_config(*, chunk_size: int) -> KBORelGNNConfig:
    return KBORelGNNConfig(
        node_feature_dims={"player": 4, "team": 8, "game": 4},
        role_feature_dims={"batting": 8, "pitching": 8},
        route_feature_dims=dict.fromkeys(KBO_TEMPORAL_ROUTE_NAMES, 6),
        hidden_dim=16,
        num_layers=2,
        num_attention_heads=4,
        dropout=0.0,
        include_boxscore_heads=True,
        route_edge_chunk_size=chunk_size,
        compact_kbo_channels=True,
    )


def test_temporal_route_family_chunking_and_empty_routes_are_lossless() -> None:
    unchunked = KBORelGNNModel(_temporal_config(chunk_size=100_000))
    chunked = KBORelGNNModel(_temporal_config(chunk_size=1))
    chunked.load_state_dict(unchunked.state_dict())
    batch = collate_kbo_day_graphs([_temporal_day()])
    first = unchunked(batch)
    second = chunked(batch)
    for name in first:
        if torch.is_tensor(first[name]):
            torch.testing.assert_close(
                first[name], second[name], rtol=3e-5, atol=3e-6, equal_nan=True
            )
        else:
            for key in first[name]:
                torch.testing.assert_close(
                    first[name][key], second[name][key], rtol=3e-5, atol=3e-6
                )
    first_loss = kbo_multitask_loss(first, batch)["loss"]
    second_loss = kbo_multitask_loss(second, batch)["loss"]
    first_loss.backward()
    second_loss.backward()
    for (first_name, first_parameter), (second_name, second_parameter) in zip(
        unchunked.named_parameters(), chunked.named_parameters(), strict=True
    ):
        assert first_name == second_name
        torch.testing.assert_close(
            first_parameter.grad,
            second_parameter.grad,
            rtol=4e-5,
            atol=4e-6,
        )

    empty = _temporal_day(23)
    for route in empty["routes"].values():
        route["source_index"] = route["source_index"][:0]
        route["destination_index"] = route["destination_index"][:0]
        route["event_features"] = route["event_features"][:0]
        route["event_age_seconds"] = route["event_age_seconds"][:0]
        route["publication_delay_seconds"] = route[
            "publication_delay_seconds"
        ][:0]
        route["weights"] = route["weights"][:0]
    empty_output = chunked(collate_kbo_day_graphs([empty]))
    assert all(
        not torch.isnan(value).any()
        for value in empty_output.values()
        if torch.is_tensor(value)
    )


def test_vnext_route_family_chunking_preserves_outputs_and_gradients() -> None:
    base = _config(vnext=True)
    unchunked = KBORelGNNModel(replace(base, route_edge_chunk_size=100_000))
    chunked = KBORelGNNModel(replace(base, route_edge_chunk_size=1))
    chunked.load_state_dict(unchunked.state_dict())
    batch = collate_kbo_day_graphs(
        [_vnext_day("2024-04-01", 31), _vnext_day("2024-04-02", 37)]
    )
    first = unchunked(batch)
    second = chunked(batch)
    for name in first:
        if torch.is_tensor(first[name]):
            torch.testing.assert_close(
                first[name], second[name], rtol=4e-5, atol=4e-6, equal_nan=True
            )
        else:
            for key in first[name]:
                torch.testing.assert_close(
                    first[name][key], second[name][key], rtol=4e-5, atol=4e-6
                )
    kbo_multitask_loss(first, batch)["loss"].backward()
    kbo_multitask_loss(second, batch)["loss"].backward()
    for (first_name, first_parameter), (second_name, second_parameter) in zip(
        unchunked.named_parameters(), chunked.named_parameters(), strict=True
    ):
        assert first_name == second_name
        torch.testing.assert_close(
            first_parameter.grad,
            second_parameter.grad,
            rtol=5e-5,
            atol=5e-6,
        )


def test_legacy_route_contract_and_v5_state_shapes_are_unchanged() -> None:
    assert KBO_ROUTE_NAMES == (
        "batter_pa_pitcher",
        "batter_participation_team",
        "pitcher_participation_team",
        "home_team_game_away_team",
    )
    assert KBO_VNEXT_ROUTE_NAMES[: len(KBO_ROUTE_NAMES)] == KBO_ROUTE_NAMES
    model = KBORelGNNModel(_config(vnext=False))
    batch = collate_kbo_day_graphs([_legacy_day()])

    assert not model.uses_game_nodes
    assert "game" not in batch["node_features"]
    assert "match_game_index" not in batch
    assert model.match_head.network[0].weight.shape == (16, 16 * 4)
    assert model.run_head.network[0].weight.shape == (32, 16 * 4)
    assert not any("node_encoders.game" in name for name in model.state_dict())
    assert torch.isfinite(model(batch)["match_logits"]).all()


def test_vnext_collator_offsets_game_queries_and_model_forward_is_finite() -> None:
    days = [_vnext_day("2024-04-01", 2), _vnext_day("2024-04-02", 3)]
    batch = collate_kbo_day_graphs(days)
    for task in ("match", "live_hit", "pa", "box_pa", "box_pitch"):
        assert batch[f"{task}_game_index"].tolist() == [1, 3]
    assert batch["node_features"]["game"].shape == (4, 4)
    assert batch["node_graph_index"]["game"].tolist() == [0, 0, 1, 1]
    assert "game_box_batting_features" not in batch
    route = next(value for value in batch["routes"] if value.route_name == "team_game_context")
    assert route.destination_index.tolist() == [1, 1, 3, 3]

    torch.manual_seed(7)
    full_config = _config(vnext=True)
    full_model = KBORelGNNModel(full_config)
    torch.manual_seed(7)
    node_only_model = KBORelGNNModel(
        replace(full_config, route_schedule=((),) * full_config.num_layers)
    )
    assert full_model.uses_game_nodes
    assert full_model.match_head.network[0].weight.shape == (16, 16 * 5)
    assert full_model.run_head.network[0].weight.shape == (32, 16 * 5)
    assert _parameter_count(full_model) > _parameter_count(node_only_model)
    assert len(node_only_model.backbone.layers) == 0
    for name, parameter in node_only_model.named_parameters():
        torch.testing.assert_close(parameter, dict(full_model.named_parameters())[name])

    output = full_model(batch)
    node_only_output = node_only_model(batch)
    for name in (
        "match_logits",
        "live_hit_joint_probabilities",
        "live_hit_expected_hits",
        "live_hit_expected_pa",
        "pa_logits",
        "box_pa_logits",
        "box_pitch_rates",
    ):
        assert torch.isfinite(output[name]).all()
        assert torch.isfinite(node_only_output[name]).all()
    for parameters in (output["match_run_parameters"], node_only_output["match_run_parameters"]):
        assert all(torch.isfinite(value).all() for value in parameters.values())

    losses = kbo_multitask_loss(output, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in full_model.parameters()
        if parameter.grad is not None
    )


def test_vnext_requires_query_game_indices_and_consistent_node_types() -> None:
    missing = _vnext_day("2024-04-01", 4)
    missing.pop("live_hit_game_index")
    with pytest.raises(ValueError, match="live_hit query index columns"):
        collate_kbo_day_graphs([missing])

    legacy = _legacy_day()
    with pytest.raises(ValueError, match="node feature keys must agree"):
        collate_kbo_day_graphs([_vnext_day("2024-04-01", 5), legacy])


def test_known_start_time_breaks_same_matchup_doubleheader_symmetry() -> None:
    day = _vnext_day("2024-04-01", 8)
    first_current = day["node_features"]["game"][1].copy()
    first_current[:] = [1, 0, 0, 0.25]
    second_current = first_current.copy()
    second_current[-1] = 0.75
    day["node_features"]["game"] = np.vstack(
        (day["node_features"]["game"][0], first_current, second_current)
    ).astype(np.float32)
    context = day["routes"]["team_game_context"]
    context["source_index"] = np.asarray([0, 1, 0, 1], dtype=np.int64)
    context["destination_index"] = np.asarray([1, 1, 2, 2], dtype=np.int64)
    context["event_features"] = np.asarray(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0]],
        dtype=np.float32,
    )
    for name in ("event_age_seconds", "publication_delay_seconds"):
        context[name] = np.zeros(4, dtype=np.float32)
    context["weights"] = np.ones(4, dtype=np.float32)
    day["match_home_team_index"] = np.asarray([0, 0], dtype=np.int64)
    day["match_away_team_index"] = np.asarray([1, 1], dtype=np.int64)
    day["match_game_index"] = np.asarray([1, 2], dtype=np.int64)
    day["match_targets"] = np.asarray([2, 2], dtype=np.int64)
    day["match_runs"] = np.asarray([[4, 2], [4, 2]], dtype=np.float32)
    day["match_query_ids"] = ("doubleheader-1", "doubleheader-2")

    torch.manual_seed(31)
    model = KBORelGNNModel(_config(vnext=True)).eval()
    batch = collate_kbo_day_graphs([day])
    with torch.no_grad():
        logits = model({**batch, "routes": ()})["match_logits"]
    assert not torch.allclose(logits[0], logits[1])
