from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    embed_dim: int = 64
    hidden_dim: int = 32
    dropout: float = 0.1


class TextAnomalyDetector(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            int(cfg.vocab_size),
            int(cfg.embed_dim),
            padding_idx=int(cfg.pad_id),
        )
        self.classifier = nn.Sequential(
            nn.Linear(int(cfg.embed_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.hidden_dim), 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = self.embedding(batch["input_ids"])
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled).squeeze(-1)


def binary_anomaly_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.to(torch.float32))


def anomaly_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) >= 0.5).to(torch.float32)
    return float((pred == labels.to(torch.float32)).to(torch.float32).mean().item())


__all__ = ["ModelConfig", "TextAnomalyDetector", "anomaly_accuracy", "binary_anomaly_loss"]
