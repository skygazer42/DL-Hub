from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 24
    num_blocks: int = 3
    num_landmarks: int = 5
    dropout: float = 0.0


class FaceLandmarkRegressor(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
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
            nn.Linear(in_ch, int(cfg.num_landmarks) * 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images.to(torch.float32))
        return torch.sigmoid(self.head(features))


def landmark_regression_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(predictions, targets)


def mean_landmark_l2_pixels(
    predictions: torch.Tensor, targets: torch.Tensor, *, image_size: int
) -> float:
    with torch.no_grad():
        pred_xy = predictions.reshape(predictions.shape[0], -1, 2) * float(image_size - 1)
        target_xy = targets.reshape(targets.shape[0], -1, 2) * float(image_size - 1)
        error = torch.linalg.vector_norm(pred_xy - target_xy, ord=2, dim=-1)
        return float(error.mean().item())


__all__ = [
    "FaceLandmarkRegressor",
    "ModelConfig",
    "landmark_regression_loss",
    "mean_landmark_l2_pixels",
]
