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
class FaceOcclusionReasoningConfig:
    vocab_size: int
    pad_id: int
    num_classes: int = 2
    hidden_dim: int = 64
    text_dim: int = 32
    vision_width: int = 32


class CompactFaceOcclusionReasoningModel(nn.Module):
    def __init__(self, cfg: FaceOcclusionReasoningConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.face_encoder = TinyVisionEncoder(vision_width=int(cfg.vision_width))
        self.query_encoder = MaskedTextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
        )
        self.query_to_vision = nn.Linear(int(cfg.text_dim), int(cfg.vision_width))
        fused_dim = int(cfg.vision_width) * 3 + int(cfg.text_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.num_classes)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image_feat = self.face_encoder(batch["image"])
        query_feat = self.query_encoder(batch["query_ids"], batch["attention_mask"])
        query_vision = self.query_to_vision(query_feat)
        feat_diff = torch.abs(image_feat - query_vision)
        feat_mul = image_feat * query_vision
        fused = torch.cat([image_feat, query_feat, feat_diff, feat_mul], dim=-1)
        logits = self.classifier(fused)
        probabilities = torch.softmax(logits, dim=-1)
        pred_labels = logits.argmax(dim=-1)
        return {"logits": logits, "probabilities": probabilities, "pred_labels": pred_labels}


def occlusion_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels.to(torch.long))


@torch.no_grad()
def occlusion_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    acc = (pred == labels.to(torch.long)).to(torch.float32).mean()
    return float(acc.item())


__all__ = [
    "FaceOcclusionReasoningConfig",
    "CompactFaceOcclusionReasoningModel",
    "occlusion_accuracy",
    "occlusion_loss",
]
