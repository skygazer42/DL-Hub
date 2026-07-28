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
class FaceParsingReasoningConfig:
    vocab_size: int
    pad_id: int
    hidden_dim: int = 64
    text_dim: int = 32
    vision_width: int = 32


class CompactFaceParsingReasoningModel(nn.Module):
    def __init__(self, cfg: FaceParsingReasoningConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.face_encoder = TinyVisionEncoder(vision_width=int(cfg.vision_width))
        self.query_encoder = MaskedTextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
        )
        self.query_proj = nn.Linear(int(cfg.text_dim), int(cfg.vision_width))
        self.mask_head = nn.Sequential(
            nn.Conv2d(int(cfg.vision_width), int(cfg.hidden_dim), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(cfg.hidden_dim), 1, kernel_size=1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image_feat = self.face_encoder(batch["image"])
        query_feat = self.query_encoder(batch["query_ids"], batch["query_mask"])
        query_gate = torch.sigmoid(self.query_proj(query_feat)).unsqueeze(-1).unsqueeze(-1)
        conditioned = image_feat * (1.0 + query_gate)
        mask_logits = self.mask_head(conditioned)
        pred_mask = torch.sigmoid(mask_logits)
        return {"mask_logits": mask_logits, "pred_mask": pred_mask}


def _soft_dice_from_probs(pred_probs: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_flat = pred_probs.flatten(start_dim=1)
    target_flat = target_mask.to(torch.float32).flatten(start_dim=1)
    inter = (pred_flat * target_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return (2.0 * inter + eps) / (denom + eps)


def face_parsing_loss(mask_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    target = target_mask.to(torch.float32)
    bce = F.binary_cross_entropy_with_logits(mask_logits, target)
    dice = _soft_dice_from_probs(torch.sigmoid(mask_logits), target).mean()
    return bce + (1.0 - dice)


@torch.no_grad()
def mask_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred_bin = (pred_mask.to(torch.float32) >= float(threshold)).to(torch.float32)
    target_bin = (target_mask.to(torch.float32) >= 0.5).to(torch.float32)
    inter = (pred_bin * target_bin).flatten(start_dim=1).sum(dim=1)
    union = (pred_bin + target_bin - pred_bin * target_bin).flatten(start_dim=1).sum(dim=1)
    return inter / union.clamp_min(1e-6)


__all__ = [
    "FaceParsingReasoningConfig",
    "CompactFaceParsingReasoningModel",
    "face_parsing_loss",
    "mask_iou",
]
