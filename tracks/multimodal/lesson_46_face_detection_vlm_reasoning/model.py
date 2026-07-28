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
class FaceDetectionReasoningConfig:
    vocab_size: int
    pad_id: int
    hidden_dim: int = 64
    text_dim: int = 32
    vision_width: int = 32


def box_iou_xyxy(pred_box: torch.Tensor, target_box: torch.Tensor) -> torch.Tensor:
    pred = pred_box.to(torch.float32)
    target = target_box.to(torch.float32)

    inter_x1 = torch.maximum(pred[:, 0], target[:, 0])
    inter_y1 = torch.maximum(pred[:, 1], target[:, 1])
    inter_x2 = torch.minimum(pred[:, 2], target[:, 2])
    inter_y2 = torch.minimum(pred[:, 3], target[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp_min(0.0)
    inter_h = (inter_y2 - inter_y1).clamp_min(0.0)
    inter = inter_w * inter_h

    area_pred = (pred[:, 2] - pred[:, 0]).clamp_min(0.0) * (pred[:, 3] - pred[:, 1]).clamp_min(0.0)
    area_target = (target[:, 2] - target[:, 0]).clamp_min(0.0) * (target[:, 3] - target[:, 1]).clamp_min(0.0)
    union = (area_pred + area_target - inter).clamp_min(1e-6)
    return inter / union


def face_detection_loss(pred_box: torch.Tensor, target_box: torch.Tensor) -> torch.Tensor:
    reg = F.smooth_l1_loss(pred_box, target_box.to(torch.float32))
    iou_term = 1.0 - box_iou_xyxy(pred_box, target_box).mean()
    return reg + 0.5 * iou_term


class CompactFaceDetectionReasoningModel(nn.Module):
    def __init__(self, cfg: FaceDetectionReasoningConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.face_encoder = TinyVisionEncoder(vision_width=int(cfg.vision_width))
        self.query_encoder = MaskedTextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
        )
        fused_dim = int(cfg.vision_width) + int(cfg.text_dim)
        self.box_head = nn.Sequential(
            nn.Linear(fused_dim, int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), 4),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image_feat = self.face_encoder(batch["image"])
        query_feat = self.query_encoder(batch["query_ids"], batch["query_mask"])
        raw = torch.sigmoid(self.box_head(torch.cat([image_feat, query_feat], dim=-1)))
        x1 = torch.minimum(raw[:, 0], raw[:, 2])
        y1 = torch.minimum(raw[:, 1], raw[:, 3])
        x2 = torch.maximum(raw[:, 0], raw[:, 2])
        y2 = torch.maximum(raw[:, 1], raw[:, 3])
        pred_box = torch.stack([x1, y1, x2, y2], dim=-1)
        return {"pred_box": pred_box}


__all__ = [
    "FaceDetectionReasoningConfig",
    "CompactFaceDetectionReasoningModel",
    "box_iou_xyxy",
    "face_detection_loss",
]
