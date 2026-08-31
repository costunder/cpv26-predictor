from __future__ import annotations

import math

import pytest

from cpv26.models import torch_available
from cpv26.models.heads import DirectPlayerGameHead, DirectRunDistributionHead

pytestmark = pytest.mark.skipif(not torch_available(), reason="PyTorch optional dependency missing")


def test_direct_player_game_probabilities_are_structurally_consistent() -> None:
    import torch

    head = DirectPlayerGameHead(
        4,
        hidden_dim=8,
        max_plate_appearances=4,
        max_hits=3,
        dropout=0.0,
    )
    head.eval()
    probabilities = head.probabilities(torch.zeros((2, 4)))

    pa = probabilities["plate_appearance_probabilities"]
    hits = probabilities["hit_count_probabilities"]
    joint = probabilities["joint_plate_appearance_hit_probabilities"]
    assert head.plate_appearance_bucket_labels == ("0", "1", "2", "3", "4", "5+")
    assert head.hit_count_bucket_labels == ("0", "1", "2", "3", "4+")
    assert pa.shape == (2, 6)
    assert hits.shape == (2, 5)
    assert joint.shape == (2, 6, 5)
    assert torch.allclose(pa.sum(dim=-1), torch.ones(2))
    assert torch.allclose(hits.sum(dim=-1), torch.ones(2))
    assert torch.allclose(
        probabilities["appearance_probability"],
        1.0 - pa[:, 0],
    )
    assert torch.count_nonzero(joint[:, 0, 1:]) == 0

    for pa_count in range(1, 5):
        for hit_count in range(4):
            if hit_count > pa_count:
                assert torch.count_nonzero(joint[:, pa_count, hit_count]) == 0
        if pa_count < 4:
            assert torch.count_nonzero(joint[:, pa_count, -1]) == 0

    support = torch.arange(5, dtype=hits.dtype)
    assert torch.allclose(
        probabilities["expected_hits"],
        (hits * support).sum(dim=-1),
    )


def test_direct_run_head_exposes_only_trained_marginal_parameters() -> None:
    import torch

    head = DirectRunDistributionHead(4, hidden_dim=8, dropout=0.0)
    output = head(torch.zeros((2, 4)), torch.zeros((2, 4)))

    assert set(output) == {
        "home_mean",
        "away_mean",
        "home_dispersion",
        "away_dispersion",
    }
    assert "score_correlation" not in output


def test_direct_player_game_joint_nll_trains_the_inference_distribution() -> None:
    import torch

    torch.manual_seed(11)
    head = DirectPlayerGameHead(
        4,
        hidden_dim=8,
        max_plate_appearances=4,
        max_hits=3,
        dropout=0.0,
    )
    embedding = torch.randn((3, 4))
    plate_appearances = torch.tensor([0, 4, 9])
    hits = torch.tensor([0, 3, 7])

    pa_bucket, hit_bucket = head.target_bucket_indices(plate_appearances, hits)
    loss = head.negative_log_likelihood(embedding, plate_appearances, hits)
    loss.backward()

    assert head.positive_plate_appearance_bucket_labels == ("1", "2", "3", "4", "5+")
    assert pa_bucket.tolist() == [0, 4, 5]
    assert hit_bucket.tolist() == [0, 3, 4]
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in head.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)

    with pytest.raises(ValueError, match="hits cannot exceed"):
        head.target_bucket_indices(torch.tensor([2]), torch.tensor([3]))


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_direct_player_game_log_nll_matches_joint_and_checkpoint(dtype_name: str) -> None:
    import torch

    dtype = getattr(torch, dtype_name)
    torch.manual_seed(31)
    head = DirectPlayerGameHead(
        4, hidden_dim=8, max_plate_appearances=4, max_hits=3, dropout=0.0
    ).to(dtype=dtype).eval()
    embedding = torch.randn((2, 3, 4), dtype=dtype)
    pa = torch.tensor([[0, 1, 2], [4, 5, 9]])
    hits = torch.tensor([[0, 1, 2], [3, 4, 7]])
    pa_bucket, hit_bucket = head.target_bucket_indices(pa, hits)
    joint = head.probabilities(embedding)["joint_plate_appearance_hit_probabilities"]
    expected = -joint.reshape(6, 6, 5)[
        torch.arange(6), pa_bucket.flatten(), hit_bucket.flatten()
    ].log().reshape(2, 3)

    losses = head.negative_log_likelihood(embedding, pa, hits, reduction="none")

    assert losses.dtype == dtype
    torch.testing.assert_close(losses, expected)
    torch.testing.assert_close(head.negative_log_likelihood(embedding, pa, hits), expected.mean())
    torch.testing.assert_close(
        head.negative_log_likelihood(embedding, pa, hits, reduction="sum"), expected.sum()
    )
    restored = DirectPlayerGameHead(
        4, hidden_dim=8, max_plate_appearances=4, max_hits=3, dropout=0.0
    ).to(dtype=dtype).eval()
    restored.load_state_dict(head.state_dict(), strict=True)
    torch.testing.assert_close(
        restored.negative_log_likelihood(embedding, pa, hits, reduction="none"), losses
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("rare_component", ["no_appearance", "appearance", "pa", "hits"])
def test_direct_player_game_extreme_nll_has_correct_nonzero_gradients(
    dtype_name: str, rare_component: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    head = DirectPlayerGameHead(
        4, hidden_dim=8, max_plate_appearances=4, max_hits=3, dropout=0.0
    )
    dtype = getattr(torch, dtype_name)
    appearance = torch.zeros(1, dtype=dtype)
    pa_logits = torch.zeros((1, 5), dtype=dtype)
    hit_logits = torch.zeros((1, 6, 5), dtype=dtype)
    pa_count, hit_count = 1, 1
    if rare_component == "no_appearance":
        appearance[0] = 20.0
        pa_count, hit_count = 0, 0
        expected = 20.0
    elif rare_component == "appearance":
        appearance[0] = -120.0
        expected = 120.0 + math.log(5) + math.log(2)
    elif rare_component == "pa":
        pa_logits[0, 0] = -120.0
        expected = math.log(2) + 120.0 + math.log(4) + math.log(2)
    else:
        pa_count, hit_count = 2, 2
        hit_logits[0, 2, 2] = -120.0
        expected = math.log(2) + math.log(5) + 120.0 + math.log(2)
    for logits in (appearance, pa_logits, hit_logits):
        logits.requires_grad_()
    # Isolate likelihood math using genuine low-precision, differentiable
    # logits; this runs on CPU CI without requiring half-precision GEMM.
    monkeypatch.setattr(head, "forward", lambda _embedding: {
        "appearance_logit": appearance,
        "plate_appearance_logits": pa_logits,
        "hit_count_logits": hit_logits,
    })
    embedding = torch.zeros((1, 4))
    pa, hits = torch.tensor([pa_count]), torch.tensor([hit_count])
    inference_joint = head.probabilities(embedding)["joint_plate_appearance_hit_probabilities"]
    assert inference_joint[0, pa_count, hit_count].item() == 0.0

    loss = head.negative_log_likelihood(embedding, pa, hits)
    loss.backward()

    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(expected, abs=2e-5)
    for logits in (appearance, pa_logits, hit_logits):
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
    if rare_component == "no_appearance":
        assert appearance.grad[0].item() == pytest.approx(1.0, abs=1e-3)
        assert torch.count_nonzero(pa_logits.grad) == 0
        assert torch.count_nonzero(hit_logits.grad) == 0
    elif rare_component == "appearance":
        assert appearance.grad[0].item() == pytest.approx(-1.0, abs=1e-3)
    elif rare_component == "pa":
        assert pa_logits.grad[0, 0].item() == pytest.approx(-1.0, abs=1e-3)
    else:
        assert hit_logits.grad[0, 2, 2].item() == pytest.approx(-1.0, abs=1e-3)


def test_direct_player_game_saturated_appearance_parameter_can_recover() -> None:
    import torch

    head = DirectPlayerGameHead(4, hidden_dim=8, dropout=0.0).eval()
    with torch.no_grad():
        head.appearance.weight.zero_()
        head.appearance.bias.fill_(20.0)
    embedding = torch.zeros((2, 4))
    pa = hits = torch.zeros(2, dtype=torch.long)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    loss = head.negative_log_likelihood(embedding, pa, hits)
    loss.backward()

    assert loss.item() == pytest.approx(20.0)
    assert head.appearance.bias.grad is not None
    assert head.appearance.bias.grad.item() == pytest.approx(1.0)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )
    optimizer.step()
    assert head.appearance.bias.item() < 20.0
    assert head.negative_log_likelihood(embedding, pa, hits).item() < loss.item()
