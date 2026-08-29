"""Plate-appearance interaction decoder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from cpv26.simulation.adapter import NEURAL_PA_OUTCOMES

from ._torch import ModuleBase, nn, require_torch

DEFAULT_PA_OUTCOMES: tuple[str, ...] = NEURAL_PA_OUTCOMES


class PlateAppearanceInteractionDecoder(ModuleBase):
    """Decode batter-pitcher context into terminal plate-appearance logits."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        context_dim: int = 0,
        hidden_dim: int = 256,
        outcomes: Sequence[str] = DEFAULT_PA_OUTCOMES,
        dropout: float = 0.1,
    ) -> None:
        require_torch()
        super().__init__()
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        if context_dim < 0:
            raise ValueError("context_dim must be non-negative")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        labels = tuple(str(value) for value in outcomes)
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("outcomes must contain at least two unique labels")

        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.outcomes = labels
        decoder_input_dim = embedding_dim * 6 + context_dim
        self.network = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(labels)),
        )

    def _optional_embedding(self, reference: Any, value: Any | None, name: str) -> Any:
        if value is None:
            return reference.new_zeros(reference.shape)
        if value.shape != reference.shape:
            raise ValueError(f"{name} must have shape {tuple(reference.shape)}")
        return value

    def forward(
        self,
        batter_embedding: Any,
        pitcher_embedding: Any,
        *,
        catcher_embedding: Any | None = None,
        defense_embedding: Any | None = None,
        game_context: Any | None = None,
    ) -> Any:
        torch, _ = require_torch()
        if batter_embedding.shape != pitcher_embedding.shape:
            raise ValueError("batter and pitcher embeddings must have identical shapes")
        if batter_embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"embedding width is {batter_embedding.shape[-1]}; expected {self.embedding_dim}"
            )
        catcher = self._optional_embedding(
            batter_embedding,
            catcher_embedding,
            "catcher_embedding",
        )
        defense = self._optional_embedding(
            batter_embedding,
            defense_embedding,
            "defense_embedding",
        )
        if game_context is None:
            game_context = batter_embedding.new_zeros(
                (*batter_embedding.shape[:-1], self.context_dim)
            )
        expected_context_shape = (*batter_embedding.shape[:-1], self.context_dim)
        if game_context.shape != expected_context_shape:
            raise ValueError(f"game_context must have shape {expected_context_shape}")

        interaction = torch.cat(
            (
                batter_embedding,
                pitcher_embedding,
                batter_embedding * pitcher_embedding,
                torch.abs(batter_embedding - pitcher_embedding),
                catcher,
                defense,
                game_context,
            ),
            dim=-1,
        )
        return self.network(interaction)

    def outcome_probabilities(self, *args: Any, **kwargs: Any) -> Any:
        torch, _ = require_torch()
        return torch.softmax(self.forward(*args, **kwargs), dim=-1)

    def outcome_index(self, label: str) -> int:
        try:
            return self.outcomes.index(label)
        except ValueError as exc:
            raise KeyError(f"unknown plate-appearance outcome: {label}") from exc


PAInteractionDecoder = PlateAppearanceInteractionDecoder
