
from dataclasses import dataclass

import torch
from torch import nn

from dlhub.vision.detection.yolo import YOLOv1Detector


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 32
    stride: int = 4


class TinyYOLOv1(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if int(cfg.stride) != 4:
            raise ValueError("This toy YOLO lesson assumes output stride=4.")

        self.model = YOLOv1Detector(
            in_channels=int(cfg.in_channels),
            num_classes=1,
            stem_channels=16,
            hidden_channels=int(cfg.hidden_channels),
            backbone_depth=1,
            head_channels=int(cfg.hidden_channels),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.model(x)
        return {"obj_logits": out["obj_logits"], "cls_logits": out["cls_logits"], "bbox": out["bbox"]}


__all__ = ["TinyYOLOv1", "ModelConfig"]

