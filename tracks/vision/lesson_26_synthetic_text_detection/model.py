from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    hidden_channels: int = 24
    num_blocks: int = 3


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class TextDetectionModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(int(cfg.in_channels), hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(int(cfg.num_blocks))])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bbox_head = nn.Linear(hidden, 4)
        self.score_head = nn.Linear(hidden, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.blocks(self.stem(images.to(torch.float32)))
        pooled = self.pool(feat).flatten(1)
        bbox = torch.sigmoid(self.bbox_head(pooled))
        score_logits = self.score_head(pooled).squeeze(-1)
        return {"bbox": bbox, "score_logits": score_logits}


def bbox_iou(pred_bbox: torch.Tensor, target_bbox: torch.Tensor) -> torch.Tensor:
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox.unbind(dim=-1)
    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = target_bbox.to(torch.float32).unbind(dim=-1)
    inter_x1 = torch.maximum(pred_x1, tgt_x1)
    inter_y1 = torch.maximum(pred_y1, tgt_y1)
    inter_x2 = torch.minimum(pred_x2, tgt_x2)
    inter_y2 = torch.minimum(pred_y2, tgt_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    pred_area = (pred_x2 - pred_x1).clamp(min=0.0) * (pred_y2 - pred_y1).clamp(min=0.0)
    tgt_area = (tgt_x2 - tgt_x1).clamp(min=0.0) * (tgt_y2 - tgt_y1).clamp(min=0.0)
    union = pred_area + tgt_area - inter_area
    return inter_area / union.clamp(min=1e-6)


def text_detection_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    target_score = targets["score"].to(torch.float32)
    score_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["score_logits"], target_score
    )

    positive_mask = target_score > 0.5
    if bool(positive_mask.any()):
        bbox_loss = torch.nn.functional.l1_loss(
            outputs["bbox"][positive_mask],
            targets["bbox"].to(torch.float32)[positive_mask],
        )
    else:
        bbox_loss = outputs["bbox"].sum() * 0.0

    total_loss = score_loss + bbox_loss
    parts = {
        "bbox_loss": float(bbox_loss.detach().item()),
        "score_loss": float(score_loss.detach().item()),
    }
    return total_loss, parts


__all__ = ["ModelConfig", "TextDetectionModel", "bbox_iou", "text_detection_loss"]
