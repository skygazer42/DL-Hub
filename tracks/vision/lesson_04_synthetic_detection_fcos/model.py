from dataclasses import dataclass

import torch
from torch import nn

from dlhub.vision.detection.fcos import FCOSDetector


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 32
    stride: int = 4


class TinyFCOS(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if int(cfg.stride) != 4:
            raise ValueError("This toy FCOS lesson assumes output stride=4.")

        # Reuse the library implementation (pure torch) while keeping the lesson API stable.
        self.model = FCOSDetector(
            in_channels=int(cfg.in_channels),
            num_classes=1,
            stem_channels=16,
            hidden_channels=int(cfg.hidden_channels),
            backbone_depth=1,
            head_convs=1,
            with_centerness=False,  # keep lesson output stable: only cls_logits + reg
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.model(x)
        return {"cls_logits": out["cls_logits"], "reg": out["reg"]}


__all__ = ["TinyFCOS", "ModelConfig"]
