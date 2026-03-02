from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    embed_dim: int = 64
    num_classes: int = 2
    dropout: float = 0.2

    # TextCNN specifics
    num_filters: int = 64
    kernel_sizes: tuple[int, ...] = (3, 4, 5)


class TextCNNClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(
            num_embeddings=int(cfg.vocab_size),
            embedding_dim=int(cfg.embed_dim),
            padding_idx=int(cfg.pad_id),
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=int(cfg.embed_dim),
                    out_channels=int(cfg.num_filters),
                    kernel_size=int(k),
                )
                for k in cfg.kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.classifier = nn.Linear(int(cfg.num_filters) * len(cfg.kernel_sizes), int(cfg.num_classes))

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)
        emb = self.embedding(input_ids)  # (B, L, D)
        x = emb.transpose(1, 2)  # (B, D, L)

        pooled: list[torch.Tensor] = []
        for conv in self.convs:
            h = torch.relu(conv(x))  # (B, C, L')
            pooled.append(torch.amax(h, dim=2))  # (B, C)

        feat = torch.cat(pooled, dim=1)
        feat = self.dropout(feat)
        return self.classifier(feat)


__all__ = ["TextCNNClassifier", "ModelConfig"]

