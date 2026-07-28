from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class TinyVisionEncoder(nn.Module):
    def __init__(self, *, vision_width: int) -> None:
        super().__init__()
        hidden = max(16, int(vision_width) // 2)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, int(vision_width), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image shape (B, C, H, W), got {tuple(image.shape)}")
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
class MultimodalReasoningConfig:
    vocab_size: int
    pad_id: int
    num_classes: int = 2
    hidden_dim: int = 64
    text_dim: int = 32
    vision_width: int = 32


class CompactMultimodalReasoningModel(nn.Module):
    def __init__(self, cfg: MultimodalReasoningConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = TinyVisionEncoder(vision_width=int(cfg.vision_width))
        self.facts_encoder = MaskedTextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
        )
        self.query_encoder = MaskedTextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
        )
        fused_dim = int(cfg.vision_width) + int(cfg.text_dim) * 2
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.num_classes)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image_features = self.vision_encoder(batch["image"])
        facts_features = self.facts_encoder(batch["facts_ids"], batch["facts_mask"])
        query_features = self.query_encoder(batch["query_ids"], batch["query_mask"])
        fused = torch.cat([image_features, facts_features, query_features], dim=-1)
        logits = self.fusion(fused)
        probs = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs}


def reasoning_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels.to(torch.long))


@torch.no_grad()
def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    acc = (preds == labels.to(torch.long)).to(torch.float32).mean()
    return float(acc.item())


__all__ = [
    "MultimodalReasoningConfig",
    "CompactMultimodalReasoningModel",
    "classification_accuracy",
    "reasoning_loss",
]
