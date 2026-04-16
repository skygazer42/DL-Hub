from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FaceDetectionConfig:
    in_channels: int = 1
    hidden_channels: int = 24
    num_blocks: int = 3
    dropout: float = 0.0


class FaceDetectionModel(nn.Module):
    def __init__(self, cfg: FaceDetectionConfig) -> None:
        super().__init__()
        in_ch = int(cfg.in_channels)
        hidden = int(cfg.hidden_channels)
        blocks: list[nn.Module] = []
        for idx in range(int(cfg.num_blocks)):
            out_ch = hidden * (2**idx)
            blocks.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_ch = out_ch

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(in_ch, 4),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.backbone(images.to(torch.float32)))
        return torch.sigmoid(raw)


def detection_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(pred_boxes, target_boxes)


def box_l1_error_pixels(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, *, image_size: int) -> float:
    with torch.no_grad():
        scale = float(max(1, image_size - 1))
        abs_err = (pred_boxes - target_boxes).abs() * scale
        return float(abs_err.mean().item())


__all__ = ["FaceDetectionConfig", "FaceDetectionModel", "box_l1_error_pixels", "detection_loss"]
