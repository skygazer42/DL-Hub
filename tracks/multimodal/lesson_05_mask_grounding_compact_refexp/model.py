from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class VisionEncoder(nn.Module):
    def __init__(self, *, vision_width: int, hidden_dim: int, mask_size: int) -> None:
        super().__init__()
        mid = max(16, int(vision_width) // 2)
        self.features = nn.Sequential(
            nn.Conv2d(3, mid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid, int(vision_width), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((int(mask_size), int(mask_size))),
        )
        self.proj = nn.Conv2d(int(vision_width), int(hidden_dim), kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.features(image.to(torch.float32))
        return self.proj(feat)


class TextEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, text_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(text_dim), padding_idx=int(pad_id))
        self.proj = nn.Linear(int(text_dim), int(hidden_dim))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids.to(torch.long))
        mask = attention_mask.to(torch.float32).unsqueeze(-1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


@dataclass(frozen=True)
class MaskGroundingModelConfig:
    vocab_size: int
    pad_id: int
    image_size: int
    mask_size: int
    hidden_dim: int = 64
    vision_width: int = 32
    text_dim: int = 32


@dataclass(frozen=True)
class MaskGroundingLossConfig:
    dice_weight: float = 1.0


class CompactMaskGroundingModel(nn.Module):
    def __init__(self, cfg: MaskGroundingModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(
            vision_width=int(cfg.vision_width),
            hidden_dim=int(cfg.hidden_dim),
            mask_size=int(cfg.mask_size),
        )
        self.text_encoder = TextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
            hidden_dim=int(cfg.hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(int(cfg.hidden_dim) * 2, int(cfg.hidden_dim), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(cfg.hidden_dim), int(cfg.hidden_dim), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mask_head = nn.Conv2d(int(cfg.hidden_dim), 1, kernel_size=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual = self.vision_encoder(batch["image"])
        text = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        text_map = text.view(int(text.shape[0]), int(text.shape[1]), 1, 1).expand_as(visual)
        fused = self.fusion(torch.cat([visual, text_map], dim=1))
        mask_logits = self.mask_head(fused)
        return {
            "mask_logits": mask_logits,
            "pred_mask": torch.sigmoid(mask_logits),
        }


def mask_grounding_loss(
    *,
    mask_logits: torch.Tensor,
    target_mask: torch.Tensor,
    cfg: MaskGroundingLossConfig,
) -> dict[str, torch.Tensor]:
    target = target_mask.to(torch.float32)
    bce_loss = F.binary_cross_entropy_with_logits(mask_logits, target)
    pred = torch.sigmoid(mask_logits)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (denom + 1e-6)).mean()
    total = bce_loss + float(cfg.dice_weight) * dice_loss
    return {"loss": total, "bce_loss": bce_loss, "dice_loss": dice_loss}


@torch.no_grad()
def mask_iou(mask_logits: torch.Tensor, target_mask: torch.Tensor, *, threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(mask_logits) >= float(threshold)).to(torch.float32)
    target = (target_mask >= 0.5).to(torch.float32)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) > 0.0).to(torch.float32).sum(dim=(1, 2, 3)).clamp_min(1.0)
    return float((intersection / union).mean().item())


@torch.no_grad()
def mask_dice_score(
    mask_logits: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    pred = (torch.sigmoid(mask_logits) >= float(threshold)).to(torch.float32)
    target = (target_mask >= 0.5).to(torch.float32)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    score = (2.0 * intersection + 1e-6) / (denom + 1e-6)
    return float(score.mean().item())


@torch.no_grad()
def foreground_accuracy(
    mask_logits: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    pred = (torch.sigmoid(mask_logits) >= float(threshold)).to(torch.float32)
    target = (target_mask >= 0.5).to(torch.float32)
    hits = (pred * target).sum(dim=(1, 2, 3))
    total = target.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return float((hits / total).mean().item())


__all__ = [
    "MaskGroundingLossConfig",
    "MaskGroundingModelConfig",
    "CompactMaskGroundingModel",
    "TextEncoder",
    "VisionEncoder",
    "foreground_accuracy",
    "mask_dice_score",
    "mask_grounding_loss",
    "mask_iou",
]
