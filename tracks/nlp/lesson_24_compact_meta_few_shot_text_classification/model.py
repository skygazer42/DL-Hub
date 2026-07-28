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
    embed_dim: int = 64
    proj_dim: int = 64
    dropout: float = 0.1


class MeanPoolEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.pad_id = int(config.pad_id)
        self.embedding = nn.Embedding(
            num_embeddings=int(config.vocab_size),
            embedding_dim=int(config.embed_dim),
            padding_idx=int(config.pad_id),
        )
        self.proj = nn.Linear(int(config.embed_dim), int(config.proj_dim))
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).to(torch.float32)
        embeddings = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        projected = self.proj(self.dropout(pooled))
        return F.normalize(projected, dim=-1)


class MetaFewShotTextClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = MeanPoolEncoder(config)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size, support_count, seq_len = batch["support_input_ids"].shape
        query_count = batch["query_input_ids"].shape[1]

        support_embeddings = self.encoder(
            batch["support_input_ids"].reshape(batch_size * support_count, seq_len),
            batch["support_attention_mask"].reshape(batch_size * support_count, seq_len),
        ).reshape(batch_size, support_count, -1)
        query_embeddings = self.encoder(
            batch["query_input_ids"].reshape(batch_size * query_count, seq_len),
            batch["query_attention_mask"].reshape(batch_size * query_count, seq_len),
        ).reshape(batch_size, query_count, -1)

        support_labels = batch["support_labels"]
        num_ways = int(support_labels.max().item()) + 1
        prototypes = []
        for class_id in range(num_ways):
            mask = (support_labels == class_id).unsqueeze(-1).to(support_embeddings.dtype)
            prototype = (support_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            prototypes.append(prototype)
        prototype_tensor = torch.stack(prototypes, dim=1)

        distances = (query_embeddings.unsqueeze(2) - prototype_tensor.unsqueeze(1)).pow(2).sum(dim=-1)
        logits = -distances
        return {
            "support_embeddings": support_embeddings,
            "query_embeddings": query_embeddings,
            "prototypes": prototype_tensor,
            "logits": logits,
        }


def meta_episode_loss(logits: torch.Tensor, query_labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), query_labels.reshape(-1))


def episode_accuracy(logits: torch.Tensor, query_labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == query_labels).to(torch.float32).mean().item())


__all__ = [
    "MetaFewShotTextClassifier",
    "ModelConfig",
    "episode_accuracy",
    "meta_episode_loss",
]
