from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    embed_dim: int = 64
    proj_dim: int = 32
    num_clusters: int = 4
    dropout: float = 0.1


class TextClusteringModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            int(cfg.vocab_size),
            int(cfg.embed_dim),
            padding_idx=int(cfg.pad_id),
        )
        self.encoder = nn.Sequential(
            nn.Linear(int(cfg.embed_dim), int(cfg.embed_dim)),
            nn.ReLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.embed_dim), int(cfg.proj_dim)),
            nn.ReLU(),
        )
        self.cluster_head = nn.Linear(int(cfg.proj_dim), int(cfg.num_clusters))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeddings = self.embedding(batch["input_ids"])
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        projected = self.encoder(pooled)
        logits = self.cluster_head(projected)
        return {"embeddings": projected, "logits": logits}


def clustering_loss(logits: torch.Tensor, cluster_labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, cluster_labels)


def cluster_accuracy(logits: torch.Tensor, cluster_labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == cluster_labels).to(torch.float32).mean().item())


__all__ = ["ModelConfig", "TextClusteringModel", "cluster_accuracy", "clustering_loss"]
