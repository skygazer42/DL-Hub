from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 24
    embedding_dim: int = 32
    dropout: float = 0.1


class ConvEmbeddingEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_channels)
        self.features = nn.Sequential(
            nn.Conv2d(int(cfg.in_channels), hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(hidden * 2, int(cfg.embedding_dim)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.head(self.features(images.to(torch.float32)))
        return F.normalize(embeddings, dim=-1)


class PrototypicalFewShotRecognizer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.encoder = ConvEmbeddingEncoder(cfg)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        support_images = batch["support_images"]
        query_images = batch["query_images"]
        support_labels = batch["support_labels"]

        batch_size, support_count, channels, height, width = support_images.shape
        query_count = int(query_images.shape[1])

        support_embeddings = self.encoder(
            support_images.reshape(batch_size * support_count, channels, height, width)
        ).reshape(batch_size, support_count, -1)
        query_embeddings = self.encoder(
            query_images.reshape(batch_size * query_count, channels, height, width)
        ).reshape(batch_size, query_count, -1)

        num_ways = int(support_labels.max().item()) + 1
        prototypes = []
        for class_id in range(num_ways):
            mask = (support_labels == class_id).unsqueeze(-1).to(support_embeddings.dtype)
            prototype = (support_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            prototypes.append(prototype)
        prototype_tensor = torch.stack(prototypes, dim=1)

        distances = (
            query_embeddings.unsqueeze(2) - prototype_tensor.unsqueeze(1)
        ).pow(2).sum(dim=-1)
        logits = -distances
        return {
            "support_embeddings": support_embeddings,
            "query_embeddings": query_embeddings,
            "prototypes": prototype_tensor,
            "logits": logits,
        }


def prototypical_loss(logits: torch.Tensor, query_labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), query_labels.reshape(-1))


def episode_accuracy(logits: torch.Tensor, query_labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == query_labels).to(torch.float32).mean().item())


__all__ = [
    "ModelConfig",
    "PrototypicalFewShotRecognizer",
    "episode_accuracy",
    "prototypical_loss",
]
