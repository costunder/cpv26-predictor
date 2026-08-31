from __future__ import annotations

import copy
from dataclasses import asdict, replace
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
    live_hit_observed_nll,
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


def _box_day(day_id: str = "2001-04-05") -> dict[str, Any]:
    day = _day(day_id)
    day["box_pa_player_index"] = np.asarray([0], dtype=np.int64)
    day["box_pa_team_index"] = np.asarray([0], dtype=np.int64)
    day["box_pa_opponent_index"] = np.asarray([1], dtype=np.int64)
    day["box_pa_counts"] = np.asarray([[2, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    day["box_pa_query_ids"] = (f"{day_id}:box-pa",)
    day["box_pitch_player_index"] = np.asarray([1], dtype=np.int64)
    day["box_pitch_team_index"] = np.asarray([1], dtype=np.int64)
    day["box_pitch_opponent_index"] = np.asarray([0], dtype=np.int64)
    day["box_pitch_targets"] = np.asarray([[5, 3, 21, 4, 1, 0, 1, 2, 1, 1]], dtype=np.float32)
    day["box_pitch_mask"] = np.ones((1, 10), dtype=bool)
    day["box_pitch_mask"][0, 2] = False
    day["box_pitch_query_ids"] = (f"{day_id}:box-pitch",)
    rng = np.random.default_rng(13)
    for kind, count in (("player", 3), ("team", 2)):
        for role, width in (("batting", 19), ("pitching", 21)):
            day[f"{kind}_box_{role}_features"] = rng.random((count, width)).astype(np.float32)
    return day


def test_unknown_pa_uses_observed_hit_and_minimum_constraint_without_fabricating_pa() -> None:
    day = _day()
    day["live_hit_pa"] = np.asarray([-1, 4], dtype=np.int64)
    day["live_hit_pa_min"] = np.asarray([3, 1], dtype=np.int64)
    day["live_hit_hits"] = np.asarray([2, 0], dtype=np.int64)
    model = _model()
    batch = collate_kbo_day_graphs([day])
    output = model(batch)
    joint = output["live_hit_joint_probabilities"]
    expected = -torch.stack((joint[0, 2:, 2].sum(), joint[1, 3, 0])).log()
    actual = live_hit_observed_nll(output["live_hit_joint_logits"], batch)
    torch.testing.assert_close(actual, expected)
    losses = kbo_multitask_loss(output, batch)
    torch.testing.assert_close(losses["live_hit_loss"], expected.mean())
    assert batch["live_hit_pa"].tolist() == [-1, 4]
    tightened = dict(batch, live_hit_pa_min=torch.tensor([6, 1]))
    assert live_hit_observed_nll(output["live_hit_joint_logits"], tightened)[0] >= actual[0]
    losses["loss"].backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_unknown_pa_minimum_above_support_uses_overflow_without_fake_exact_count() -> None:
    day = _day()
    day["live_hit_pa"] = np.asarray([-1, -1], dtype=np.int64)
    day["live_hit_pa_min"] = np.asarray([12, 1], dtype=np.int64)
    day["live_hit_hits"] = np.asarray([9, 0], dtype=np.int64)
    batch = collate_kbo_day_graphs([day])
    output = _model()(batch)
    losses = live_hit_observed_nll(output["live_hit_joint_logits"], batch)
    joint = output["live_hit_joint_probabilities"]
    torch.testing.assert_close(losses[0], -joint[0, -1, -1].log())
    torch.testing.assert_close(losses[1], -joint[1, :, 0].sum().log())
    assert batch["live_hit_pa"].tolist() == [-1, -1]
    assert batch["live_hit_pa_min"].tolist() == [12, 1]


@pytest.mark.parametrize(
    ("pa", "minimum", "hits"), [(0, 1, 0), (-2, 1, 0), (-1, 0, 0), (2, 3, 1), (2, 1, 3)]
)
def test_partial_pa_collator_rejects_invalid_evidence(pa: int, minimum: int, hits: int) -> None:
    day = _day()
    day["live_hit_pa"] = np.asarray([pa, 3], dtype=np.int64)
    day["live_hit_pa_min"] = np.asarray([minimum, 1], dtype=np.int64)
    day["live_hit_hits"] = np.asarray([hits, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="Live Hit labels"):
        collate_kbo_day_graphs([day])


def test_boxscore_heads_train_verified_histograms_and_masked_pitch_counts() -> None:
    day = _box_day()
    config = replace(_config(), include_boxscore_heads=True)
    model = KBORelGNNModel(config)
    batch = collate_kbo_day_graphs([day])
    output = model(batch)
    assert output["box_pa_logits"].shape == (1, 10)
    assert output["box_pitch_rates"].shape == (1, 10)
    assert batch["box_pitch_targets"][0, 2] == 0
    losses = kbo_multitask_loss(output, batch)
    histogram_nll = -(
        batch["box_pa_counts"] * torch.log_softmax(output["box_pa_logits"], dim=-1)
    ).sum() / 4
    torch.testing.assert_close(losses["box_pa_loss"], histogram_nll)
    mask = batch["box_pitch_mask"]
    rates = output["box_pitch_rates"]
    target = batch["box_pitch_targets"]
    pitch_nll = rates - target * rates.log() + torch.lgamma(target + 1)
    torch.testing.assert_close(losses["box_pitch_loss"], pitch_nll[mask].mean())
    changed = copy.deepcopy(day)
    changed["box_pitch_targets"][0, 2] = 900
    changed_batch = collate_kbo_day_graphs([changed])
    assert torch.equal(batch["box_pitch_targets"], changed_batch["box_pitch_targets"])
    aggregate_loss = losses["box_pa_loss"] + losses["box_pitch_loss"]
    aggregate_loss.backward()
    for head in (model.box_pa_head, model.box_pitch_head):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in head.parameters())
    gradient = model.backbone.player_encoder.shared_core[0].weight.grad
    assert gradient is not None and gradient[:, -40:].abs().sum() > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)


def test_boxscore_and_legacy_days_collate_without_cross_graph_offsets_or_feature_leakage() -> None:
    first, second = _day("2023-04-01"), _box_day()
    model = KBORelGNNModel(replace(_config(), include_boxscore_heads=True)).eval()
    union = collate_kbo_day_graphs([first, second], max_pa_per_day=0,
                                  max_edges_per_route_per_day=0)
    assert union["box_pa_player_index"].tolist() == [3]
    assert union["box_pitch_player_index"].tolist() == [4]
    assert union["box_pa_team_index"].tolist() == [2]
    assert union["box_pitch_opponent_index"].tolist() == [2]
    output = model(union)
    separate = [model(collate_kbo_day_graphs([day])) for day in (first, second)]
    for key in ("match_logits", "box_pa_logits", "box_pitch_rates"):
        torch.testing.assert_close(output[key], torch.cat([part[key] for part in separate]),
                                   atol=2e-6, rtol=1e-5)
    changed = copy.deepcopy(second)
    changed["player_box_batting_features"] *= 100
    altered = model(collate_kbo_day_graphs([first, changed]))
    torch.testing.assert_close(output["match_logits"][:1], altered["match_logits"][:1])
    with pytest.raises(ValueError, match="include_boxscore_heads"):
        _model()(collate_kbo_day_graphs([second]))


@pytest.mark.parametrize("limit", [None, 0])
def test_unlimited_pa_and_edge_mode_retains_every_query_and_relation(limit: int | None) -> None:
    batch = collate_kbo_day_graphs([_day()], max_pa_per_day=limit,
                                  max_edges_per_route_per_day=limit)
    assert len(batch["pa_targets"]) == 6
    assert [route.num_edges for route in batch["routes"]] == [3, 2, 1, 2]


def test_boxscore_evaluation_separates_aggregates_and_masks_unknown_pa_from_mae() -> None:
    from cpv26.training.kbo_runner import KBOTrainingConfig, _evaluate_model

    day = _box_day()
    day["live_hit_pa"] = np.asarray([-1, 4], dtype=np.int64)
    day["live_hit_pa_min"] = np.asarray([12, 1], dtype=np.int64)
    model = KBORelGNNModel(replace(_config(), include_boxscore_heads=True)).eval()
    batch = collate_kbo_day_graphs([day])
    output = model(batch)
    expected_pa_mae = abs(float(output["live_hit_expected_pa"][1].detach()) - 4)
    report, predictions = _evaluate_model(
        model, [batch], KBOTrainingConfig(device="cpu"), torch.device("cpu"), None,
        collect_predictions=True,
    )
    live = report["live_hit"]
    assert live["known_pa_samples"] == live["unknown_pa_samples"] == 1
    assert live["unknown_pa_minimum_overflow_samples"] == 1
    assert live["expected_pa_lower_bound_mae"] == pytest.approx(expected_pa_mae)
    assert np.isfinite(live["partial_pa_nll"])
    assert predictions["live_hit"][0]["observed_pa"] is None
    assert predictions["live_hit"][0]["observed_pa_lower_bound"] == 12
    assert predictions["live_hit"][1]["observed_pa"] == 4
    assert report["box_pa"]["observed_outcomes"] == 4
    assert report["box_pa"]["player_game_queries"] == 1
    assert report["box_pitch"]["observed_counts"] == 9
    assert report["box_pitch"]["per_field"]["pitches_thrown"]["samples"] == 0
    assert report["box_pitch"]["per_field"]["pitches_thrown"]["mae"] is None
    assert predictions["box_pitch"][0]["observed_pitches_thrown"] is None
    assert len(predictions["box_pa"]) == len(predictions["box_pitch"]) == 1
    assert np.isfinite(report["selection_loss"])


def test_all_unknown_pa_evaluation_has_no_pa_mae_or_joint_nll() -> None:
    from cpv26.training.kbo_runner import KBOTrainingConfig, _evaluate_model

    day = _day()
    day["live_hit_pa"] = np.asarray([-1, -1], dtype=np.int64)
    batch = collate_kbo_day_graphs([day])
    report, _ = _evaluate_model(
        _model(), [batch], KBOTrainingConfig(device="cpu"), torch.device("cpu"), None
    )
    assert report["live_hit"]["samples"] == 2
    assert report["live_hit"]["expected_pa_lower_bound_mae"] is None
    assert report["live_hit"]["joint_nll"] is None
    assert report["live_hit"]["unknown_pa_samples"] == 2


def test_boxscore_targets_cannot_change_pregame_predictions() -> None:
    original = _box_day()
    changed = copy.deepcopy(original)
    changed["box_pa_counts"] *= 2
    changed["box_pitch_targets"] *= 3
    model = KBORelGNNModel(replace(_config(), include_boxscore_heads=True)).eval()
    first = model(collate_kbo_day_graphs([original]))
    second = model(collate_kbo_day_graphs([changed]))
    for key in ("match_logits", "live_hit_joint_probabilities", "box_pa_logits", "box_pitch_rates"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_boxscore_config_default_keeps_old_state_dict_compatible() -> None:
    legacy = _config().to_dict()
    for key in (
        "include_boxscore_heads", "box_batting_feature_dim", "box_pitching_feature_dim",
        "box_gradient_mode",
    ):
        legacy.pop(key)
    restored = KBORelGNNConfig(**legacy)
    assert restored.include_boxscore_heads is False
    assert restored.box_gradient_mode == "shared"
    source = _model()
    target = KBORelGNNModel(restored)
    target.load_state_dict(source.state_dict(), strict=True)
    assert all(
        not key.startswith(("box_pa_head.", "box_pitch_head.")) for key in source.state_dict()
    )


@pytest.mark.parametrize("mode", ["shared", "head_only"])
def test_box_gradient_policy_controls_only_aggregate_to_backbone_gradients(mode: str) -> None:
    model = KBORelGNNModel(replace(
        _config(), include_boxscore_heads=True, box_gradient_mode=mode,
    ))
    batch = collate_kbo_day_graphs([_box_day()])
    losses = kbo_multitask_loss(model(batch), batch)
    (losses["box_pa_loss"] + losses["box_pitch_loss"]).backward()
    backbone_has_gradient = any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in model.backbone.parameters()
    )
    assert backbone_has_gradient is (mode == "shared")
    for head in (model.box_pa_head, model.box_pitch_head):
        assert any(
            parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
            for parameter in head.parameters()
        )
    assert all(parameter.grad is None for parameter in model.match_head.parameters())


def test_head_only_policy_preserves_primary_updates_even_when_box_gradients_are_large() -> None:
    from cpv26.training.kbo_runner import _clip_gradient_norms

    torch.manual_seed(19)
    original = KBORelGNNModel(replace(
        _config(), include_boxscore_heads=True, box_gradient_mode="head_only",
    ))
    ordinary = [_box_day("2001-04-05"), _box_day("2023-04-01")]
    changed = copy.deepcopy(ordinary)
    for day in changed:
        day["box_pa_counts"] = np.roll(day["box_pa_counts"], 4, axis=1)
        day["box_pitch_targets"] *= 3
    states, gradients, norms = [], [], []
    for days, weight in ((ordinary, 0.0), (changed, 1000.0)):
        model = copy.deepcopy(original)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0)
        batch = collate_kbo_day_graphs(days)
        loss = kbo_multitask_loss(
            model(batch), batch, run_weight=0.1, box_pa_weight=weight, box_pitch_weight=weight,
        )["loss"]
        loss.backward()
        norms.append(_clip_gradient_norms(model, 0.01))
        gradients.append({
            name: parameter.grad.clone() if parameter.grad is not None else None
            for name, parameter in model.named_parameters()
        })
        optimizer.step()
        states.append(model.state_dict())
    assert set(norms[1]) == {"primary", "box_heads"}
    assert norms[1]["primary"] > 0.01 and norms[1]["box_heads"] > 0.01
    for name in gradients[0]:
        if name.startswith(("box_pa_head.", "box_pitch_head.")):
            continue
        if gradients[0][name] is None:
            assert gradients[1][name] is None
        else:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
    assert any(
        not torch.equal(states[0][name], states[1][name])
        for name in states[0] if name.startswith(("box_pa_head.", "box_pitch_head."))
    )


def test_legacy_decoder_mixed_partial_pa_predictions_have_consistent_columns() -> None:
    from cpv26.training.kbo_runner import KBOTrainingConfig, _evaluate_model

    known = _day("2023-04-01")
    partial = _day("2001-04-05")
    partial["live_hit_pa"] = np.asarray([-1, 3], dtype=np.int64)
    partial["live_hit_pa_min"] = np.asarray([2, 3], dtype=np.int64)
    batches = [collate_kbo_day_graphs([day]) for day in (known, partial)]
    _, predictions = _evaluate_model(
        _model(), batches, KBOTrainingConfig(device="cpu"), torch.device("cpu"), None,
        collect_predictions=True,
    )
    rows = predictions["live_hit"]
    assert all(set(row) == set(rows[0]) for row in rows)
    assert [row["observed_pa"] for row in rows] == [4, 3, None, 3]
    assert [row["observed_pa_lower_bound"] for row in rows] == [4, 3, 2, 3]


@pytest.mark.parametrize("with_history", [False, True])
@pytest.mark.parametrize("include_boxscore", [False, True])
def test_game_only_day_without_players_trains_match_and_run_without_optional_labels(
    with_history: bool,
    include_boxscore: bool,
) -> None:
    model = KBORelGNNModel(replace(_config(), include_boxscore_heads=include_boxscore))
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
    if include_boxscore:
        assert output["box_pa_logits"].shape == output["box_pitch_rates"].shape == (0, 10)
        assert losses["box_pa_loss"].item() == losses["box_pitch_loss"].item() == 0
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
