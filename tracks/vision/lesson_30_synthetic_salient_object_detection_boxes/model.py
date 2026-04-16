from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
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


class SalientObjectBoxesModel(nn.Module):
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
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(images.to(torch.float32)))
        return self.head(self.pool(features))


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    boxes = boxes.to(torch.float32)
    cx, cy, w, h = boxes.unbind(dim=-1)
    half_w = 0.5 * w.clamp(0.0, 1.0)
    half_h = 0.5 * h.clamp(0.0, 1.0)
    x1 = (cx - half_w).clamp(0.0, 1.0)
    y1 = (cy - half_h).clamp(0.0, 1.0)
    x2 = (cx + half_w).clamp(0.0, 1.0)
    y2 = (cy + half_h).clamp(0.0, 1.0)
    return torch.stack((x1, y1, x2, y2), dim=-1)


def _pairwise_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    pred_xyxy = _cxcywh_to_xyxy(pred_boxes)
    target_xyxy = _cxcywh_to_xyxy(target_boxes)

    inter_x1 = torch.maximum(pred_xyxy[..., 0], target_xyxy[..., 0])
    inter_y1 = torch.maximum(pred_xyxy[..., 1], target_xyxy[..., 1])
    inter_x2 = torch.minimum(pred_xyxy[..., 2], target_xyxy[..., 2])
    inter_y2 = torch.minimum(pred_xyxy[..., 3], target_xyxy[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp_min(0.0)
    inter_h = (inter_y2 - inter_y1).clamp_min(0.0)
    inter_area = inter_w * inter_h

    pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp_min(0.0) * (
        pred_xyxy[..., 3] - pred_xyxy[..., 1]
    ).clamp_min(0.0)
    target_area = (target_xyxy[..., 2] - target_xyxy[..., 0]).clamp_min(0.0) * (
        target_xyxy[..., 3] - target_xyxy[..., 1]
    ).clamp_min(0.0)
    union = (pred_area + target_area - inter_area).clamp_min(1e-6)
    return inter_area / union


def box_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> float:
    return float(_pairwise_iou(pred_boxes, target_boxes).mean().item())


def salient_box_loss(
    pred_boxes: torch.Tensor, target_boxes: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    target_boxes = target_boxes.to(torch.float32)
    pred_boxes = pred_boxes.to(torch.float32)
    l1_loss = torch.nn.functional.l1_loss(pred_boxes, target_boxes)
    iou_loss = 1.0 - _pairwise_iou(pred_boxes, target_boxes).mean()
    total_loss = l1_loss + iou_loss
    return total_loss, {"l1_loss": float(l1_loss.item()), "iou_loss": float(iou_loss.item())}


__all__ = ["ModelConfig", "SalientObjectBoxesModel", "box_iou", "salient_box_loss"]
