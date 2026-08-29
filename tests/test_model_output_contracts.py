from __future__ import annotations

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
