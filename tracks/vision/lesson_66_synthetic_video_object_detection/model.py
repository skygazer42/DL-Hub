from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 16
    num_blocks: int = 3
    max_objects: int = 2
    num_classes: int = 3


class VideoObjectDetectionModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = max(8, int(cfg.hidden_channels))
        blocks = max(1, int(cfg.num_blocks))
        layers: list[nn.Module] = [
            nn.Conv3d(int(cfg.in_channels), hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks - 1):
            layers.extend(
                [
                    nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm3d(hidden),
                    nn.ReLU(inplace=True),
                ]
            )
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        out_dim = int(cfg.max_objects) * (4 + 1 + int(cfg.num_classes))
        self.head = nn.Linear(hidden, out_dim)
        self.max_objects = int(cfg.max_objects)
        self.num_classes = int(cfg.num_classes)

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        x = clip.to(torch.float32)
        if x.ndim != 5:
            raise ValueError(f"Expected (B,T,C,H,W), got {tuple(x.shape)}")
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        feat = self.encoder(x)
        pooled = self.pool(feat).flatten(1)
        raw = self.head(pooled).view(x.shape[0], self.max_objects, 5 + self.num_classes)
        pred_boxes = torch.sigmoid(raw[..., :4])
        pred_scores = torch.sigmoid(raw[..., 4])
        pred_logits = raw[..., 5:]
        return {"pred_boxes": pred_boxes, "pred_scores": pred_scores, "pred_logits": pred_logits}


def video_object_detection_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    target_boxes = targets["boxes"].to(torch.float32)
    target_scores = targets["present"].to(torch.float32)
    target_labels = targets["labels"].to(torch.long)

    box_err = torch.nn.functional.smooth_l1_loss(outputs["pred_boxes"], target_boxes, reduction="none").mean(dim=-1)
    denom = target_scores.sum().clamp(min=1.0)
    box_loss = (box_err * target_scores).sum() / denom
    score_loss = torch.nn.functional.binary_cross_entropy(outputs["pred_scores"], target_scores)

    class_loss = torch.tensor(0.0, device=outputs["pred_logits"].device)
    pos = target_scores > 0.5
    if bool(pos.any()):
        class_loss = torch.nn.functional.cross_entropy(outputs["pred_logits"][pos], target_labels[pos])

    total = box_loss + score_loss + class_loss
    parts = {
        "box_loss": float(box_loss.item()),
        "score_loss": float(score_loss.item()),
        "class_loss": float(class_loss.item()),
    }
    return total, parts


__all__ = ["ModelConfig", "VideoObjectDetectionModel", "video_object_detection_loss"]
