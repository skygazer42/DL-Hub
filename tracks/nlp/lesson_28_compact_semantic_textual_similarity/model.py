from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    embed_dim: int = 64
    hidden_dim: int = 96
    dropout: float = 0.1


class SemanticTextualSimilarityRegressor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=int(config.vocab_size),
            embedding_dim=int(config.embed_dim),
            padding_idx=int(config.pad_id),
        )
        self.projection = nn.Sequential(
            nn.Linear(int(config.embed_dim), int(config.hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.hidden_dim), 1),
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.config.pad_id)).to(torch.float32)

        embeddings = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        score = self.projection(pooled).squeeze(-1)
        return torch.sigmoid(score)


def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        return float(torch.mean(torch.abs(predictions - targets)).item())


__all__ = ["ModelConfig", "SemanticTextualSimilarityRegressor", "mean_absolute_error"]
