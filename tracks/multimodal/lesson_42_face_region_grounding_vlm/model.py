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
class FaceRegionGroundingConfig:
    vocab_size: int
    pad_id: int
    hidden_dim: int = 64
    text_dim: int = 32
    vision_width: int = 32


class ToyFaceRegionGroundingModel(nn.Module):
    def __init__(self, cfg: FaceRegionGroundingConfig) -> None:
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
        raw_boxes = torch.sigmoid(self.box_head(torch.cat([image_feat, query_feat], dim=-1)))

        x_sorted = torch.sort(raw_boxes[:, [0, 2]], dim=1).values
        y_sorted = torch.sort(raw_boxes[:, [1, 3]], dim=1).values
        pred_boxes = torch.stack([x_sorted[:, 0], y_sorted[:, 0], x_sorted[:, 1], y_sorted[:, 1]], dim=1)
        return {"pred_boxes": pred_boxes}


def _box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    w = (boxes[:, 2] - boxes[:, 0]).clamp_min(0.0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
    return w * h


def box_iou_xyxy(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    inter_x1 = torch.maximum(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.maximum(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.minimum(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.minimum(pred_boxes[:, 3], target_boxes[:, 3])
    inter = _box_area_xyxy(torch.stack([inter_x1, inter_y1, inter_x2, inter_y2], dim=1))
    union = _box_area_xyxy(pred_boxes) + _box_area_xyxy(target_boxes) - inter
    return inter / union.clamp_min(1e-6)


def face_region_grounding_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    reg = F.smooth_l1_loss(pred_boxes, target_boxes.to(torch.float32))
    iou_term = 1.0 - box_iou_xyxy(pred_boxes, target_boxes.to(torch.float32)).mean()
    return reg + iou_term


__all__ = [
    "FaceRegionGroundingConfig",
    "ToyFaceRegionGroundingModel",
    "box_iou_xyxy",
    "face_region_grounding_loss",
]
