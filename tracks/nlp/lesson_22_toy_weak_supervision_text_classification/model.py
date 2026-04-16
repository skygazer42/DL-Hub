from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    embed_dim: int = 24
    hidden_dim: int = 16
    num_labeling_functions: int = 3
    dropout: float = 0.1


class WeakSupervisionTextClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(
            num_embeddings=int(cfg.vocab_size),
            embedding_dim=int(cfg.embed_dim),
            padding_idx=int(cfg.pad_id),
        )
        lf_dim = int(cfg.num_labeling_functions) * 2
        self.fusion = nn.Sequential(
            nn.Linear(int(cfg.embed_dim) + lf_dim, int(cfg.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.hidden_dim), 2),
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        lf_votes = batch["lf_votes"].to(torch.float32)
        lf_mask = batch["lf_mask"].to(torch.float32)

        emb = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(emb.dtype)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        vote_values = (lf_votes * 2.0 - 1.0) * lf_mask
        lf_features = torch.cat([vote_values, lf_mask], dim=1)
        return self.fusion(torch.cat([pooled, lf_features], dim=1))


def weak_supervision_loss(logits: torch.Tensor, label_probs: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -(label_probs * log_probs).sum(dim=1).mean()


def weak_supervision_accuracy(logits: torch.Tensor, gold_labels: torch.Tensor) -> float:
    with torch.no_grad():
        return float((logits.argmax(dim=1) == gold_labels).to(torch.float32).mean().item())


__all__ = [
    "ModelConfig",
    "WeakSupervisionTextClassifier",
    "weak_supervision_accuracy",
    "weak_supervision_loss",
]
