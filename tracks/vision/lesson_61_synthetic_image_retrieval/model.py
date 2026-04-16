from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 24
    num_blocks: int = 3
    embedding_dim: int = 32
    dropout: float = 0.0


class ImageRetrievalEmbeddingNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_ch = int(cfg.in_channels)
        hidden = int(cfg.hidden_channels)
        blocks: list[nn.Module] = []
        for idx in range(int(cfg.num_blocks)):
            out_ch = hidden * (2**idx)
            blocks.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_ch = out_ch

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(in_ch, int(cfg.embedding_dim)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.head(self.backbone(images.to(torch.float32)))
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def triplet_margin_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    distances = torch.cdist(embeddings, embeddings, p=2.0)
    total_loss = embeddings.new_tensor(0.0)
    valid_triplets = 0

    for anchor_idx in range(int(embeddings.shape[0])):
        anchor_label = labels[anchor_idx]
        positive_mask = labels == anchor_label
        positive_mask[anchor_idx] = False
        negative_mask = labels != anchor_label
        if not torch.any(positive_mask) or not torch.any(negative_mask):
            continue

        pos_dist = distances[anchor_idx][positive_mask].mean()
        neg_dist = distances[anchor_idx][negative_mask].min()
        total_loss = total_loss + torch.relu(pos_dist - neg_dist + float(margin))
        valid_triplets += 1

    if valid_triplets == 0:
        return embeddings.sum() * 0.0
    return total_loss / float(valid_triplets)


def retrieval_top1_accuracy(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    if int(embeddings.shape[0]) < 2:
        return 0.0
    with torch.no_grad():
        scores = embeddings @ embeddings.t()
        scores.fill_diagonal_(-float("inf"))
        nn_index = scores.argmax(dim=1)
        return float((labels[nn_index] == labels).to(torch.float32).mean().item())


__all__ = [
    "ImageRetrievalEmbeddingNet",
    "ModelConfig",
    "retrieval_top1_accuracy",
    "triplet_margin_loss",
]
