from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cpv26.models.kbo_relgnn import (  # noqa: E402
    KBO_ROUTE_NAMES,
    KBORelGNNConfig,
    KBORelGNNModel,
    collate_kbo_day_graphs,
    encode_live_hit_targets,
    kbo_multitask_loss,
)


def _config() -> KBORelGNNConfig:
    return KBORelGNNConfig(
        node_feature_dims={"player": 4, "team": 8},
        role_feature_dims={"batting": 8, "pitching": 8},
        route_feature_dims=dict.fromkeys(KBO_ROUTE_NAMES, 6),
        hidden_dim=16,
        num_layers=2,
        num_attention_heads=4,
        dropout=0.0,
    )


def _day(day_id: str = "2024-04-01", seed: int = 1) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    endpoints = {
        "batter_pa_pitcher": ([0, 2, 0], [1, 1, 1]),
        "batter_participation_team": ([0, 2], [0, 0]),
        "pitcher_participation_team": ([1], [1]),
        "home_team_game_away_team": ([0, 1], [1, 0]),
    }
    routes = {}
    for name, (source, destination) in endpoints.items():
        count = len(source)
        routes[name] = {
            "source_index": np.asarray(source, dtype=np.int64),
            "destination_index": np.asarray(destination, dtype=np.int64),
            "event_features": rng.normal(size=(count, 6)).astype(np.float32),
            "event_age_seconds": np.asarray(
                [86_400 * (index + 1) for index in range(count)], dtype=np.float32
            ),
            "publication_delay_seconds": np.full(count, 3_600, dtype=np.float32),
            "weights": np.linspace(0.3, 1.0, count, dtype=np.float32),
        }
    return {
        "day_id": day_id,
        "node_features": {
            "player": rng.normal(size=(3, 4)).astype(np.float32),
            "team": rng.normal(size=(2, 8)).astype(np.float32),
        },
        "role_features": {
            "batting": rng.normal(size=(3, 8)).astype(np.float32),
            "pitching": rng.normal(size=(3, 8)).astype(np.float32),
        },
        "routes": routes,
        "match_home_team_index": np.asarray([0], dtype=np.int64),
        "match_away_team_index": np.asarray([1], dtype=np.int64),
        "match_targets": np.asarray([2], dtype=np.int64),
        "match_runs": np.asarray([[5, 3]], dtype=np.float32),
        "match_query_ids": (f"{day_id}:g1",),
        "live_hit_player_index": np.asarray([0, 2], dtype=np.int64),
        "live_hit_team_index": np.asarray([0, 0], dtype=np.int64),
        "live_hit_opponent_index": np.asarray([1, 1], dtype=np.int64),
        "live_hit_pa": np.asarray([4, 3], dtype=np.int64),
        "live_hit_hits": np.asarray([2, 0], dtype=np.int64),
        "live_hit_query_ids": (f"{day_id}:g1:p0", f"{day_id}:g1:p2"),
        "pa_batter_index": np.asarray([0, 2, 0, 2, 0, 2], dtype=np.int64),
        "pa_pitcher_index": np.asarray([1, 1, 1, 1, 1, 1], dtype=np.int64),
        "pa_targets": np.asarray([2, 0, 6, 1, 5, 8], dtype=np.int64),
        "pa_context": rng.normal(size=(6, 10)).astype(np.float32),
        "pa_query_ids": tuple(f"{day_id}:pa:{index}" for index in range(6)),
    }


def _model() -> Any:
    torch.manual_seed(31)
    return KBORelGNNModel(_config())


def _game_only_day(day_id: str = "2001-04-05", *, with_history: bool = True) -> dict[str, Any]:
    day = _day(day_id)
    day["node_features"]["player"] = np.empty((0, 4), dtype=np.float32)
    for role in day["role_features"]:
        day["role_features"][role] = np.empty((0, 8), dtype=np.float32)
    for name, route in day["routes"].items():
        if not with_history or name != "home_team_game_away_team":
            for key, value in route.items():
                route[key] = value[:0]
    if not with_history:
        day["node_features"]["team"][:] = 0
    for key in tuple(day):
        if key.startswith(("live_hit_", "pa_")):
            day[key] = day[key][:0]
    # Explicit score labels remain present when the source supplies no PA records.
    day["match_runs"] = np.asarray([[11, 6]], dtype=np.float32)
    return day


@pytest.mark.parametrize("with_history", [False, True])
def test_game_only_day_without_players_trains_match_and_run_without_optional_labels(
    with_history: bool,
) -> None:
    model = _model()
    batch = collate_kbo_day_graphs([_game_only_day(with_history=with_history)])
    assert batch["node_features"]["player"].shape == (0, 4)
    assert batch["node_graph_index"]["player"].numel() == 0
    assert batch["match_runs"].tolist() == [[11, 6]]
    output = model(batch)
    assert output["match_logits"].shape == (1, 3)
    assert output["live_hit_joint_probabilities"].shape == (0, 9, 7)
    assert output["pa_logits"].shape == (0, 10)
    losses = kbo_multitask_loss(output, batch, run_weight=0.1)
    assert all(torch.isfinite(value) for value in losses.values())
    assert losses["live_hit_loss"].item() == 0
    assert losses["pa_loss"].item() == 0
    assert losses["match_loss"].item() > 0
    assert losses["run_loss"].item() > 0
    losses["loss"].backward()
    assert all(parameter.grad is None for parameter in model.live_hit_head.parameters())
    assert all(parameter.grad is None for parameter in model.pa_head.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.run_head.parameters()
    )
    if with_history:
        gradient = (
            model.backbone.layers[0].messages["home_team_game_away_team"].forward_value.weight.grad
        )
        assert gradient is not None and gradient.abs().sum() > 0
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


@pytest.mark.parametrize("historical_first", [False, True])
def test_game_only_and_pa_days_share_disjoint_batches_without_label_or_offset_leakage(
    historical_first: bool,
) -> None:
    historical, recent = _game_only_day(), _day("2023-04-01")
    days = [historical, recent] if historical_first else [recent, historical]
    model = _model().eval()
    separate = [model(collate_kbo_day_graphs([day])) for day in days]
    batch = collate_kbo_day_graphs(days)
    output = model(batch)
    assert batch["day_ids"] == tuple(day["day_id"] for day in days)
    assert batch["match_graph_index"].tolist() == [0, 1]
    assert batch["live_hit_graph_index"].tolist() == [int(historical_first)] * 2
    assert batch["pa_graph_index"].tolist() == [int(historical_first)] * 6
    assert batch["live_hit_player_index"].tolist() == [0, 2]
    assert batch["match_home_team_index"].tolist() == [0, 2]
    for key in ("match_logits", "live_hit_joint_probabilities", "pa_logits"):
        assert torch.allclose(output[key], torch.cat([item[key] for item in separate]), atol=2e-6)
    losses = kbo_multitask_loss(output, batch, run_weight=0.1)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_evaluation_of_only_game_history_reports_missing_pa_tasks_without_fake_metrics() -> None:
    from cpv26.training.kbo_runner import KBOTrainingConfig, _evaluate_model

    batches = [
        collate_kbo_day_graphs([_game_only_day("2001-04-05", with_history=False)]),
        collate_kbo_day_graphs([_game_only_day("2001-04-06")]),
    ]
    report, predictions = _evaluate_model(
        _model(),
        batches,
        KBOTrainingConfig(device="cpu"),
        torch.device("cpu"),
        None,
        collect_predictions=True,
    )
    assert report["match"]["samples"] == 2
    assert report["live_hit"] is None
    assert report["pa"] is None
    assert report["losses"]["live_hit"] == report["losses"]["pa"] == 0
    assert np.isfinite(report["selection_loss"])
    assert len(predictions["match"]) == 2
    assert predictions["live_hit"] == predictions["pa"] == []


def test_actual_relgnn_backpropagates_all_task_losses_through_graph_relations() -> None:
    model = _model()
    batch = collate_kbo_day_graphs([_day()])
    output = model(batch)
    losses = kbo_multitask_loss(output, batch, run_weight=0.1)

    assert output["match_logits"].shape == (1, 3)
    assert output["live_hit_joint_logits"].shape == (2, 9, 7)
    assert output["pa_logits"].shape == (6, 10)
    assert set(losses) == {"loss", "match_loss", "live_hit_loss", "pa_loss", "run_loss"}
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    for route_name in (
        "batter_pa_pitcher",
        "batter_participation_team",
        "pitcher_participation_team",
    ):
        weight = model.backbone.layers[0].messages[route_name].forward_value.weight
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0
    core_gradient = model.backbone.player_encoder.shared_core[0].weight.grad
    assert core_gradient is not None and core_gradient.abs().sum() > 0
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_live_hit_inference_and_nll_share_one_masked_positive_pa_joint() -> None:
    model = _model().eval()
    batch = collate_kbo_day_graphs([_day()])
    output = model(batch)
    joint = output["live_hit_joint_probabilities"]
    assert torch.allclose(joint.sum(dim=(1, 2)), torch.ones(2))
    assert torch.equal(
        joint[:, ~model.joint_allowed], torch.zeros_like(joint[:, ~model.joint_allowed])
    )
    assert torch.allclose(output["live_hit_hit_probability"], 1 - joint[:, :, 0].sum(dim=1))
    assert torch.allclose(
        output["live_hit_expected_hits"], (joint * model.hit_support).sum(dim=(1, 2))
    )
    assert torch.allclose(
        output["live_hit_expected_pa"], (joint * model.pa_support[:, None]).sum(dim=(1, 2))
    )
    assert torch.all(output["live_hit_expected_hits"] <= output["live_hit_expected_pa"])
    target = encode_live_hit_targets(batch["live_hit_pa"], batch["live_hit_hits"], _config())
    selected = joint.flatten(1)[torch.arange(2), target]
    assert torch.allclose(
        kbo_multitask_loss(output, batch)["live_hit_loss"], -selected.log().mean()
    )
    assert "appearance_probability" not in output


def test_overflow_target_encoding_and_invalid_nonappearance_labels() -> None:
    encoded = encode_live_hit_targets(torch.tensor([1, 8, 12]), torch.tensor([0, 5, 9]), _config())
    assert encoded.tolist() == [0, 7 * 7 + 5, 8 * 7 + 6]
    with pytest.raises(ValueError, match="PA >= 1"):
        encode_live_hit_targets(torch.tensor([0]), torch.tensor([0]), _config())
    bad = _day()
    bad["live_hit_pa"][0] = 0
    with pytest.raises(ValueError, match="PA >= 1"):
        collate_kbo_day_graphs([bad])


def test_disjoint_union_offsets_and_predictions_match_individual_days() -> None:
    first, second = _day(), _day("2024-04-02", 9)
    model = _model().eval()
    separate = [model(collate_kbo_day_graphs([day])) for day in (first, second)]
    union = collate_kbo_day_graphs([first, second])
    combined = model(union)

    assert union["match_home_team_index"].tolist() == [0, 2]
    assert union["live_hit_player_index"].tolist() == [0, 2, 3, 5]
    assert union["pa_pitcher_index"].tolist() == [1] * 6 + [4] * 6
    assert union["match_graph_index"].tolist() == [0, 1]
    for key in ("match_logits", "live_hit_joint_probabilities", "pa_logits"):
        assert torch.allclose(combined[key], torch.cat([item[key] for item in separate]), atol=2e-6)
    changed = copy.deepcopy(second)
    changed["node_features"]["player"] *= 100
    changed_output = model(collate_kbo_day_graphs([first, changed]))
    assert torch.allclose(
        changed_output["match_logits"][:1], combined["match_logits"][:1], atol=2e-6
    )


def test_current_query_lineup_identity_or_labels_do_not_change_match_prediction() -> None:
    model = _model().eval()
    original = _day()
    changed = copy.deepcopy(original)
    changed["live_hit_player_index"] = np.asarray([2, 0], dtype=np.int64)
    changed["live_hit_hits"] = np.asarray([0, 3], dtype=np.int64)
    changed["pa_batter_index"][:] = 1
    changed["pa_pitcher_index"][:] = 0
    original_output = model(collate_kbo_day_graphs([original]))
    changed_output = model(collate_kbo_day_graphs([changed]))
    assert torch.equal(original_output["match_logits"], changed_output["match_logits"])


@pytest.mark.parametrize("query_count", [0, 1, 12])
def test_auxiliary_pa_context_and_query_count_cannot_change_pregame_outputs(
    query_count: int,
) -> None:
    model = _model().eval()
    original = _day()
    changed = copy.deepcopy(original)
    changed["pa_batter_index"] = np.full(query_count, 2, dtype=np.int64)
    changed["pa_pitcher_index"] = np.full(query_count, 0, dtype=np.int64)
    changed["pa_targets"] = np.full(query_count, 9, dtype=np.int64)
    changed["pa_context"] = np.full((query_count, 10), 1234, dtype=np.float32)
    changed["pa_query_ids"] = tuple(f"changed:{index}" for index in range(query_count))
    original_output = model(collate_kbo_day_graphs([original]))
    changed_output = model(collate_kbo_day_graphs([changed]))
    for key in ("match_logits", "live_hit_joint_probabilities", "live_hit_hit_probability"):
        assert torch.equal(original_output[key], changed_output[key])


def test_empty_routes_and_empty_optional_pa_queries_are_finite() -> None:
    day = _day()
    for route in day["routes"].values():
        for key, array in route.items():
            route[key] = array[:0]
    for key in ("pa_batter_index", "pa_pitcher_index", "pa_targets", "pa_context"):
        day[key] = day[key][:0]
    day["pa_query_ids"] = ()
    batch = collate_kbo_day_graphs([day])
    output = _model()(batch)
    losses = kbo_multitask_loss(output, batch)
    assert output["pa_logits"].shape == (0, 10)
    assert losses["pa_loss"].item() == 0
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


def test_pa_sampling_and_edge_cap_are_per_day_deterministic() -> None:
    first, second = _day(), _day("2024-04-02", 8)
    single = collate_kbo_day_graphs(
        [first], max_pa_per_day=3, seed=5, max_edges_per_route_per_day=1
    )
    union = collate_kbo_day_graphs(
        [second, first], max_pa_per_day=3, seed=5, max_edges_per_route_per_day=1
    )
    assert single["pa_query_ids"] == union["pa_query_ids"][3:]
    assert all(route.num_edges == 1 for route in single["routes"])
    assert all(route.event_age_seconds.tolist() == [86_400] for route in single["routes"])
    assert single["pa_context"].shape == (3, 10)


def test_collator_rejects_future_or_cross_day_invalid_edges_and_model_requires_validation() -> None:
    future = _day()
    future["routes"]["batter_pa_pitcher"]["publication_delay_seconds"][0] = 1e9
    with pytest.raises(ValueError, match="after its cutoff"):
        collate_kbo_day_graphs([future])
    invalid = _day()
    invalid["routes"]["batter_pa_pitcher"]["source_index"][0] = 3
    with pytest.raises(IndexError, match="node range"):
        collate_kbo_day_graphs([invalid, _day("2024-04-02")])
    with pytest.raises(ValueError, match="collate"):
        _model()({})


def test_cpu_bfloat16_autocast_keeps_attention_accumulators_and_losses_finite() -> None:
    model = _model()
    batch = collate_kbo_day_graphs([_day()])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch)
        losses = kbo_multitask_loss(output, batch, run_weight=0.1)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_zero_weight_routes_remain_finite_under_autocast() -> None:
    day = _day()
    for route in day["routes"].values():
        route["weights"][:] = 0
    model = _model()
    batch = collate_kbo_day_graphs([day])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = kbo_multitask_loss(model(batch), batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime is unavailable")
def test_cuda_float16_amp_batch_device_and_graph_gradients() -> None:
    model = _model().to("cuda")
    batch = collate_kbo_day_graphs([_day()], device="cuda")
    assert all(value.device.type == "cuda" for value in batch["node_features"].values())
    for route in batch["routes"]:
        assert all(
            value.device.type == "cuda"
            for value in asdict(route).values()
            if torch.is_tensor(value)
        )
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(batch)
        loss = kbo_multitask_loss(output, batch, run_weight=0.1)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_config_serialization_roundtrips_plain_safe_checkpoint_values() -> None:
    config = _config()
    assert KBORelGNNConfig(**config.to_dict()).to_dict() == config.to_dict()
