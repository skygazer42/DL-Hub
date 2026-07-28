from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class RegionInteractionEncoder(nn.Module):
    def __init__(self, *, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.region_proj = nn.Linear(int(feature_dim) + 4, int(hidden_dim))
        self.out_proj = nn.Linear(int(hidden_dim), int(hidden_dim))

    def forward(self, region_features: torch.Tensor, region_boxes: torch.Tensor) -> torch.Tensor:
        if region_features.ndim != 3:
            raise ValueError(
                f"Expected region_features shape (B, R, D), got {tuple(region_features.shape)}"
            )
        if region_boxes.ndim != 3:
            raise ValueError(f"Expected region_boxes shape (B, R, 4), got {tuple(region_boxes.shape)}")
        fused = torch.cat([region_features.to(torch.float32), region_boxes.to(torch.float32)], dim=-1)
        hidden = torch.relu(self.region_proj(fused))
        pooled = hidden.mean(dim=1)
        return self.out_proj(pooled)


class MaskedTextEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, text_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(text_dim), padding_idx=int(pad_id))
        self.proj = nn.Linear(int(text_dim), int(text_dim))

    def forward(self, query_ids: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        token_embed = self.embedding(query_ids.to(torch.long))
        mask = query_mask.to(torch.float32).unsqueeze(-1)
        pooled = (token_embed * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


@dataclass(frozen=True)
class HoiReasoningConfig:
    vocab_size: int
    pad_id: int
    num_classes: int = 2
    feature_dim: int = 16
    text_dim: int = 32
    hidden_dim: int = 48


class CompactHoiReasoningModel(nn.Module):
    def __init__(self, cfg: HoiReasoningConfig) -> None:
        super().__init__()
        self.region_encoder = RegionInteractionEncoder(
            feature_dim=int(cfg.feature_dim),
            hidden_dim=int(cfg.hidden_dim),
        )
        self.text_encoder = MaskedTextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(int(cfg.hidden_dim + cfg.text_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.num_classes)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        region_state = self.region_encoder(batch["region_features"], batch["region_boxes"])
        query_state = self.text_encoder(batch["query_ids"], batch["query_mask"])
        fused = torch.cat([region_state, query_state], dim=-1)
        logits = self.classifier(fused)
        probabilities = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probabilities": probabilities}


def hoi_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels.to(torch.long))


@torch.no_grad()
def hoi_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    acc = (pred == labels.to(torch.long)).to(torch.float32).mean()
    return float(acc.item())


__all__ = [
    "HoiReasoningConfig",
    "MaskedTextEncoder",
    "RegionInteractionEncoder",
    "CompactHoiReasoningModel",
    "hoi_accuracy",
    "hoi_loss",
]
