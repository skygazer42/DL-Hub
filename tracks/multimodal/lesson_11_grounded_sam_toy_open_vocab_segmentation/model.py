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


class PromptEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, text_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(text_dim), padding_idx=int(pad_id))
        self.proj = nn.Linear(int(text_dim), int(hidden_dim))

    def forward(self, query_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(query_ids.to(torch.long))
        mask = attention_mask.to(torch.float32).unsqueeze(-1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


class MaskDecoder(nn.Module):
    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(int(hidden_dim) * 2, int(hidden_dim), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(hidden_dim), int(hidden_dim), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mask_head = nn.Conv2d(int(hidden_dim), 1, kernel_size=1)

    def forward(self, visual: torch.Tensor, prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_map = prompt.view(int(prompt.shape[0]), int(prompt.shape[1]), 1, 1).expand_as(visual)
        fused = self.fusion(torch.cat([visual, prompt_map], dim=1))
        mask_logits = self.mask_head(fused)
        return fused, mask_logits


@dataclass(frozen=True)
class GroundedSamModelConfig:
    vocab_size: int
    pad_id: int
    image_size: int
    mask_size: int
    hidden_dim: int = 64
    vision_width: int = 32
    text_dim: int = 32


@dataclass(frozen=True)
class GroundedSamLossConfig:
    dice_weight: float = 1.0


class ToyGroundedSamModel(nn.Module):
    def __init__(self, cfg: GroundedSamModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(
            vision_width=int(cfg.vision_width),
            hidden_dim=int(cfg.hidden_dim),
            mask_size=int(cfg.mask_size),
        )
        self.prompt_encoder = PromptEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
            hidden_dim=int(cfg.hidden_dim),
        )
        self.mask_decoder = MaskDecoder(hidden_dim=int(cfg.hidden_dim))
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(int(cfg.hidden_dim), 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual = self.vision_encoder(batch["image"])
        prompt = self.prompt_encoder(batch["query_ids"], batch["attention_mask"])
        fused, mask_logits = self.mask_decoder(visual, prompt)
        presence_logit = self.presence_head(fused).squeeze(-1)
        return {
            "presence_logit": presence_logit,
            "mask_logits": mask_logits,
            "pred_mask": torch.sigmoid(mask_logits),
        }


def grounded_sam_loss(
    *,
    presence_logit: torch.Tensor,
    mask_logits: torch.Tensor,
    target_present: torch.Tensor,
    target_mask: torch.Tensor,
    cfg: GroundedSamLossConfig,
) -> dict[str, torch.Tensor]:
    target_present = target_present.to(torch.float32)
    presence_loss = F.binary_cross_entropy_with_logits(presence_logit, target_present)

    positive_mask = target_present > 0.5
    if bool(positive_mask.any()):
        pos_mask_logits = mask_logits[positive_mask]
        pos_target_mask = target_mask[positive_mask].to(torch.float32)
        mask_bce_loss = F.binary_cross_entropy_with_logits(pos_mask_logits, pos_target_mask)

        pred = torch.sigmoid(pos_mask_logits)
        intersection = (pred * pos_target_mask).sum(dim=(1, 2, 3))
        denom = pred.sum(dim=(1, 2, 3)) + pos_target_mask.sum(dim=(1, 2, 3))
        mask_dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (denom + 1e-6)).mean()
    else:
        zero = presence_loss.new_zeros(())
        mask_bce_loss = zero
        mask_dice_loss = zero

    total = presence_loss + mask_bce_loss + float(cfg.dice_weight) * mask_dice_loss
    return {
        "loss": total,
        "presence_loss": presence_loss,
        "mask_bce_loss": mask_bce_loss,
        "mask_dice_loss": mask_dice_loss,
    }


@torch.no_grad()
def presence_accuracy(presence_logit: torch.Tensor, target_present: torch.Tensor) -> float:
    pred = (torch.sigmoid(presence_logit) >= 0.5).to(torch.float32)
    target = target_present.to(torch.float32)
    return float((pred == target).to(torch.float32).mean().item())


@torch.no_grad()
def mask_iou(
    mask_logits: torch.Tensor,
    target_mask: torch.Tensor,
    target_present: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    positive_mask = target_present > 0.5
    if not bool(positive_mask.any()):
        return 0.0
    pred = (torch.sigmoid(mask_logits[positive_mask]) >= float(threshold)).to(torch.float32)
    target = (target_mask[positive_mask] >= 0.5).to(torch.float32)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) > 0.0).to(torch.float32).sum(dim=(1, 2, 3)).clamp_min(1.0)
    return float((intersection / union).mean().item())


@torch.no_grad()
def mask_dice_score(
    mask_logits: torch.Tensor,
    target_mask: torch.Tensor,
    target_present: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    positive_mask = target_present > 0.5
    if not bool(positive_mask.any()):
        return 0.0
    pred = (torch.sigmoid(mask_logits[positive_mask]) >= float(threshold)).to(torch.float32)
    target = (target_mask[positive_mask] >= 0.5).to(torch.float32)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    score = (2.0 * intersection + 1e-6) / (denom + 1e-6)
    return float(score.mean().item())


@torch.no_grad()
def foreground_accuracy(
    mask_logits: torch.Tensor,
    target_mask: torch.Tensor,
    target_present: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> float:
    positive_mask = target_present > 0.5
    if not bool(positive_mask.any()):
        return 0.0
    pred = (torch.sigmoid(mask_logits[positive_mask]) >= float(threshold)).to(torch.float32)
    target = (target_mask[positive_mask] >= 0.5).to(torch.float32)
    hits = (pred * target).sum(dim=(1, 2, 3))
    total = target.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return float((hits / total).mean().item())


__all__ = [
    "GroundedSamLossConfig",
    "GroundedSamModelConfig",
    "MaskDecoder",
    "PromptEncoder",
    "ToyGroundedSamModel",
    "VisionEncoder",
    "foreground_accuracy",
    "grounded_sam_loss",
    "mask_dice_score",
    "mask_iou",
    "presence_accuracy",
]
