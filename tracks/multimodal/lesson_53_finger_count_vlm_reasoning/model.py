from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class TinyVisionEncoder(nn.Module):
    def __init__(self, *, vision_width: int) -> None:
        super().__init__()
        hidden = max(20, int(vision_width) // 2)
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, int(vision_width), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image.to(torch.float32))


class MaskedTextEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, text_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(text_dim), padding_idx=int(pad_id))
        self.proj = nn.Linear(int(text_dim), int(text_dim))

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        token_embed = self.embedding(token_ids.to(torch.long))
        mask = token_mask.to(torch.float32).unsqueeze(-1)
        pooled = (token_embed * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


@dataclass(frozen=True)
class FingerCountReasoningConfig:
    vocab_size: int
    pad_id: int
    num_classes: int = 6
    hidden_dim: int = 72
    text_dim: int = 32
    vision_width: int = 40


class ToyFingerCountReasoningModel(nn.Module):
    def __init__(self, cfg: FingerCountReasoningConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision = TinyVisionEncoder(vision_width=int(cfg.vision_width))
        self.text = MaskedTextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
        )
        fused_dim = int(cfg.vision_width) + int(cfg.text_dim)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.num_classes)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image_feat = self.vision(batch["image"])
        query_feat = self.text(batch["query_ids"], batch["query_mask"])
        logits = self.head(torch.cat([image_feat, query_feat], dim=-1))
        return {"logits": logits}


def finger_count_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target.to(torch.long))


def compute_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return float((preds == target.to(torch.long)).to(torch.float32).mean().item())


__all__ = [
    "FingerCountReasoningConfig",
    "ToyFingerCountReasoningModel",
    "compute_accuracy",
    "finger_count_loss",
]

