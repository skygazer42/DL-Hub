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
class FaceAlignmentReasoningConfig:
    vocab_size: int
    pad_id: int
    hidden_dim: int = 64
    text_dim: int = 32
    vision_width: int = 32
    num_points: int = 5


class CompactFaceAlignmentReasoningModel(nn.Module):
    def __init__(self, cfg: FaceAlignmentReasoningConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.face_encoder = TinyVisionEncoder(vision_width=int(cfg.vision_width))
        self.query_encoder = MaskedTextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
        )
        fused_dim = int(cfg.vision_width) + int(cfg.text_dim)
        self.point_head = nn.Sequential(
            nn.Linear(fused_dim, int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.num_points) * 2),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image_feat = self.face_encoder(batch["image"])
        query_feat = self.query_encoder(batch["query_ids"], batch["query_mask"])
        pred_points = torch.sigmoid(self.point_head(torch.cat([image_feat, query_feat], dim=-1)))
        pred_points = pred_points.reshape(-1, int(self.cfg.num_points), 2)
        return {"pred_points": pred_points}


def mean_alignment_l2(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    delta = pred_points - target_points.to(torch.float32)
    return torch.linalg.norm(delta, dim=-1).mean(dim=1)


def face_alignment_loss(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    reg = F.smooth_l1_loss(pred_points, target_points.to(torch.float32))
    point_l2 = mean_alignment_l2(pred_points, target_points).mean()
    return reg + 0.25 * point_l2


__all__ = [
    "FaceAlignmentReasoningConfig",
    "CompactFaceAlignmentReasoningModel",
    "face_alignment_loss",
    "mean_alignment_l2",
]
