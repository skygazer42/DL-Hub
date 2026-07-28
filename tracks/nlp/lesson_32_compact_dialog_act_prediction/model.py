from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    num_acts: int = 6
    embed_dim: int = 64
    dropout: float = 0.1


class DialogActPredictor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=int(config.vocab_size),
            embedding_dim=int(config.embed_dim),
            padding_idx=int(config.pad_id),
        )
        self.dropout = nn.Dropout(float(config.dropout))
        self.classifier = nn.Linear(int(config.embed_dim), int(config.num_acts))

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.config.pad_id)).to(torch.float32)

        token_embeddings = self.dropout(self.embedding(input_ids))
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def dialog_act_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return float((preds == labels).to(torch.float32).mean().item())


__all__ = ["DialogActPredictor", "ModelConfig", "compute_accuracy", "dialog_act_loss"]
