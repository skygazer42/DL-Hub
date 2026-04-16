from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.vision.action_recognition.c3d import build_c3d_video_classifier


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    num_classes: int = 4
    backbone_variant: str = "c3d_tiny"
    backbone_width_mult: float = 0.5
    dropout: float = 0.1


class ActionRecognitionModel(nn.Module):
    """Tiny action recognizer backed by a local C3D family model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.backbone = build_c3d_video_classifier(
            in_channels=int(cfg.in_channels),
            num_classes=int(cfg.num_classes),
            variant=str(cfg.backbone_variant),
            width_mult=float(cfg.backbone_width_mult),
            dropout=float(cfg.dropout),
            frames=8,
            image_size=32,
        )

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        x = clip.to(torch.float32)
        if x.ndim != 5:
            raise ValueError(f"Expected (B,T,C,H,W), got {tuple(x.shape)}")
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return {"action_logits": self.backbone(x)}


def action_recognition_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    action_loss = torch.nn.functional.cross_entropy(
        outputs["action_logits"],
        targets["action_label"].to(torch.long),
    )
    return action_loss, {"action_loss": float(action_loss.item())}


def action_recognition_accuracy(action_logits: torch.Tensor, action_label: torch.Tensor) -> float:
    pred = action_logits.argmax(dim=-1)
    return float((pred == action_label.to(torch.long)).to(torch.float32).mean().item())


__all__ = [
    "ActionRecognitionModel",
    "ModelConfig",
    "action_recognition_accuracy",
    "action_recognition_loss",
]

