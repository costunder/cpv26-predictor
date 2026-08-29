"""Direct player-game, team run, and win/draw/loss prediction heads."""

from __future__ import annotations

from typing import Any

from ._torch import ModuleBase, nn, require_torch


class DirectPlayerGameHead(ModuleBase):
    """Predict a constrained joint PA/hit distribution for a player-game.

    The final bucket of each distribution is an explicit overflow bucket. Its
    lower bound is used as the support value when deriving ``expected_hits``.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        max_plate_appearances: int = 8,
        max_hits: int = 5,
        dropout: float = 0.1,
    ) -> None:
        require_torch()
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if max_plate_appearances < 1 or max_hits < 1:
            raise ValueError("count maxima must be positive")
        if max_hits > max_plate_appearances:
            raise ValueError("max_hits cannot exceed max_plate_appearances")
        self.input_dim = input_dim
        self.max_plate_appearances = max_plate_appearances
        self.max_hits = max_hits
        self.plate_appearance_bucket_labels = (
            "0",
            *(str(value) for value in range(1, max_plate_appearances + 1)),
            f"{max_plate_appearances + 1}+",
        )
        self.hit_count_bucket_labels = (
            *(str(value) for value in range(max_hits + 1)),
            f"{max_hits + 1}+",
        )
        self.positive_plate_appearance_bucket_labels = (
            *self.plate_appearance_bucket_labels[1:],
        )
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.appearance = nn.Linear(hidden_dim, 1)
        self.plate_appearances = nn.Linear(hidden_dim, max_plate_appearances + 1)
        # Each PA bucket receives its own hit-count logits.  The structural
        # mask below still owns support validation (H <= PA), while this
        # parameterization lets the model learn genuinely different
        # P(H | PA, x) distributions instead of reusing one marginal vector
        # and changing only its normalization mask.
        self.hits = nn.Linear(
            hidden_dim,
            (max_plate_appearances + 2) * (max_hits + 2),
        )

    def _structured_probabilities(
        self,
        appearance_logit: Any,
        plate_appearance_logits: Any,
        hit_count_logits: Any,
    ) -> dict[str, Any]:
        torch, _ = require_torch()
        appearance_probability = torch.sigmoid(appearance_logit)
        positive_pa_probabilities = torch.softmax(plate_appearance_logits, dim=-1)
        plate_appearance_probabilities = torch.cat(
            (
                (1.0 - appearance_probability).unsqueeze(-1),
                appearance_probability.unsqueeze(-1) * positive_pa_probabilities,
            ),
            dim=-1,
        )

        pa_bucket_count = self.max_plate_appearances + 2
        hit_bucket_count = self.max_hits + 2
        allowed = torch.zeros(
            (pa_bucket_count, hit_bucket_count),
            dtype=torch.bool,
            device=hit_count_logits.device,
        )
        allowed[0, 0] = True
        for plate_appearances in range(1, self.max_plate_appearances + 1):
            allowed[plate_appearances, : min(plate_appearances, self.max_hits) + 1] = True
            if plate_appearances >= self.max_hits + 1:
                allowed[plate_appearances, -1] = True
        allowed[-1, :] = True

        expected_hit_logit_shape = (
            *plate_appearance_logits.shape[:-1],
            pa_bucket_count,
            hit_bucket_count,
        )
        if hit_count_logits.shape != expected_hit_logit_shape:
            raise ValueError(
                "hit_count_logits must have shape "
                f"{expected_hit_logit_shape}; received {tuple(hit_count_logits.shape)}"
            )
        masked_hit_logits = hit_count_logits.masked_fill(
            ~allowed,
            torch.finfo(hit_count_logits.dtype).min,
        )
        conditional_hit_probabilities = torch.softmax(masked_hit_logits, dim=-1)
        joint_probabilities = (
            plate_appearance_probabilities.unsqueeze(-1)
            * conditional_hit_probabilities
        )
        hit_count_probabilities = joint_probabilities.sum(dim=-2)
        hit_support = torch.arange(
            hit_bucket_count,
            dtype=hit_count_probabilities.dtype,
            device=hit_count_probabilities.device,
        )
        expected_hits = (hit_count_probabilities * hit_support).sum(dim=-1)
        return {
            "appearance_probability": appearance_probability,
            "plate_appearance_probabilities": plate_appearance_probabilities,
            "conditional_hit_probabilities": conditional_hit_probabilities,
            "joint_plate_appearance_hit_probabilities": joint_probabilities,
            "hit_count_probabilities": hit_count_probabilities,
            "expected_hits": expected_hits,
        }

    def forward(self, candidate_embedding: Any) -> dict[str, Any]:
        require_torch()
        if candidate_embedding.shape[-1] != self.input_dim:
            raise ValueError(
                f"candidate embedding width is {candidate_embedding.shape[-1]}; "
                f"expected {self.input_dim}"
            )
        hidden = self.trunk(candidate_embedding)
        appearance_logit = self.appearance(hidden).squeeze(-1)
        plate_appearance_logits = self.plate_appearances(hidden)
        hit_count_logits = self.hits(hidden).reshape(
            *hidden.shape[:-1],
            self.max_plate_appearances + 2,
            self.max_hits + 2,
        )
        structured = self._structured_probabilities(
            appearance_logit,
            plate_appearance_logits,
            hit_count_logits,
        )
        return {
            "appearance_logit": appearance_logit,
            "plate_appearance_logits": plate_appearance_logits,
            "hit_count_logits": hit_count_logits,
            "expected_hits": structured["expected_hits"],
        }

    def probabilities(self, candidate_embedding: Any) -> dict[str, Any]:
        require_torch()
        output = self.forward(candidate_embedding)
        return self._structured_probabilities(
            output["appearance_logit"],
            output["plate_appearance_logits"],
            output["hit_count_logits"],
        )

    def target_bucket_indices(
        self,
        plate_appearances: Any,
        hits: Any,
    ) -> tuple[Any, Any]:
        """Encode integer PA/hit targets into the declared overflow buckets."""

        torch, _ = require_torch()
        if plate_appearances.shape != hits.shape:
            raise ValueError("plate_appearances and hits targets must have identical shapes")
        if plate_appearances.dtype == torch.bool or hits.dtype == torch.bool:
            raise TypeError("plate_appearances and hits targets must be integer tensors")
        if plate_appearances.is_floating_point() or hits.is_floating_point():
            raise TypeError("plate_appearances and hits targets must be integer tensors")
        if bool((plate_appearances < 0).any().item()) or bool((hits < 0).any().item()):
            raise ValueError("plate_appearances and hits targets must be non-negative")
        if bool((hits > plate_appearances).any().item()):
            raise ValueError("hits cannot exceed plate appearances")
        pa_bucket = plate_appearances.to(dtype=torch.long).clamp_max(
            self.max_plate_appearances + 1
        )
        hit_bucket = hits.to(dtype=torch.long).clamp_max(self.max_hits + 1)
        return pa_bucket, hit_bucket

    def negative_log_likelihood(
        self,
        candidate_embedding: Any,
        plate_appearances: Any,
        hits: Any,
        *,
        reduction: str = "mean",
    ) -> Any:
        """Train the same constrained joint PA/hit distribution used at inference."""

        torch, _ = require_torch()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        probabilities = self.probabilities(candidate_embedding)
        joint = probabilities["joint_plate_appearance_hit_probabilities"]
        pa_bucket, hit_bucket = self.target_bucket_indices(plate_appearances, hits)
        pa_bucket = pa_bucket.to(device=joint.device)
        hit_bucket = hit_bucket.to(device=joint.device)
        expected_target_shape = joint.shape[:-2]
        if pa_bucket.shape != expected_target_shape:
            raise ValueError(f"target tensors must have shape {tuple(expected_target_shape)}")
        flat_joint = joint.reshape(-1, joint.shape[-2], joint.shape[-1])
        flat_pa = pa_bucket.reshape(-1)
        flat_hits = hit_bucket.reshape(-1)
        row_index = torch.arange(flat_joint.shape[0], device=joint.device)
        selected = flat_joint[row_index, flat_pa, flat_hits]
        losses = -torch.log(selected.clamp_min(torch.finfo(joint.dtype).tiny)).reshape(
            expected_target_shape
        )
        if reduction == "none":
            return losses
        if reduction == "sum":
            return losses.sum()
        return losses.mean()


class DirectRunDistributionHead(ModuleBase):
    """Predict independent home/away negative-binomial marginals."""

    def __init__(
        self,
        team_embedding_dim: int,
        *,
        context_dim: int = 0,
        hidden_dim: int = 192,
        dropout: float = 0.1,
        minimum_dispersion: float = 0.05,
    ) -> None:
        require_torch()
        super().__init__()
        if team_embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("team_embedding_dim and hidden_dim must be positive")
        if context_dim < 0 or minimum_dispersion <= 0:
            raise ValueError("context_dim must be non-negative and dispersion positive")
        self.team_embedding_dim = team_embedding_dim
        self.context_dim = context_dim
        self.minimum_dispersion = minimum_dispersion
        input_dim = team_embedding_dim * 4 + context_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        home_embedding: Any,
        away_embedding: Any,
        *,
        game_context: Any | None = None,
    ) -> dict[str, Any]:
        torch, _ = require_torch()
        if home_embedding.shape != away_embedding.shape:
            raise ValueError("home and away embeddings must have identical shapes")
        if home_embedding.shape[-1] != self.team_embedding_dim:
            raise ValueError(
                f"team embedding width is {home_embedding.shape[-1]}; "
                f"expected {self.team_embedding_dim}"
            )
        expected_context_shape = (*home_embedding.shape[:-1], self.context_dim)
        if game_context is None:
            game_context = home_embedding.new_zeros(expected_context_shape)
        if game_context.shape != expected_context_shape:
            raise ValueError(f"game_context must have shape {expected_context_shape}")
        features = torch.cat(
            (
                home_embedding,
                away_embedding,
                home_embedding - away_embedding,
                home_embedding * away_embedding,
                game_context,
            ),
            dim=-1,
        )
        raw = self.network(features)
        return {
            "home_mean": torch.nn.functional.softplus(raw[..., 0]) + 1e-4,
            "away_mean": torch.nn.functional.softplus(raw[..., 1]) + 1e-4,
            "home_dispersion": (
                torch.nn.functional.softplus(raw[..., 2]) + self.minimum_dispersion
            ),
            "away_dispersion": (
                torch.nn.functional.softplus(raw[..., 3]) + self.minimum_dispersion
            ),
        }

    @staticmethod
    def negative_binomial_log_prob(
        observed_runs: Any,
        mean: Any,
        dispersion: Any,
    ) -> Any:
        """Log probability under an NB2 parameterization."""

        torch, _ = require_torch()
        observed = observed_runs.to(dtype=mean.dtype)
        mean = mean.clamp_min(1e-8)
        dispersion = dispersion.clamp_min(1e-8)
        return (
            torch.lgamma(observed + dispersion)
            - torch.lgamma(dispersion)
            - torch.lgamma(observed + 1.0)
            + dispersion * (torch.log(dispersion) - torch.log(dispersion + mean))
            + observed * (torch.log(mean) - torch.log(dispersion + mean))
        )


class WDLHead(ModuleBase):
    """Direct away-win/draw/home-win classification head."""

    classes: tuple[str, str, str] = ("away_win", "draw", "home_win")

    def __init__(
        self,
        team_embedding_dim: int,
        *,
        context_dim: int = 0,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        require_torch()
        super().__init__()
        if team_embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("team_embedding_dim and hidden_dim must be positive")
        if context_dim < 0:
            raise ValueError("context_dim must be non-negative")
        self.team_embedding_dim = team_embedding_dim
        self.context_dim = context_dim
        self.network = nn.Sequential(
            nn.Linear(team_embedding_dim * 4 + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        home_embedding: Any,
        away_embedding: Any,
        *,
        game_context: Any | None = None,
    ) -> Any:
        torch, _ = require_torch()
        if home_embedding.shape != away_embedding.shape:
            raise ValueError("home and away embeddings must have identical shapes")
        if home_embedding.shape[-1] != self.team_embedding_dim:
            raise ValueError(
                f"team embedding width is {home_embedding.shape[-1]}; "
                f"expected {self.team_embedding_dim}"
            )
        expected_context_shape = (*home_embedding.shape[:-1], self.context_dim)
        if game_context is None:
            game_context = home_embedding.new_zeros(expected_context_shape)
        if game_context.shape != expected_context_shape:
            raise ValueError(f"game_context must have shape {expected_context_shape}")
        features = torch.cat(
            (
                home_embedding,
                away_embedding,
                home_embedding - away_embedding,
                home_embedding * away_embedding,
                game_context,
            ),
            dim=-1,
        )
        return self.network(features)

    def probabilities(self, *args: Any, **kwargs: Any) -> Any:
        torch, _ = require_torch()
        return torch.softmax(self.forward(*args, **kwargs), dim=-1)


RunDistributionHead = DirectRunDistributionHead
