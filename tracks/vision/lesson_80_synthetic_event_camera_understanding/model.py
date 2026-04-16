from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

import dlhub.vision.event_camera_understanding as event_model_zoo


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 5
    hidden_channels: int = 16
    family: str = "ev_cnn"
    variant: str = "ev_cnn_tiny"
    width_mult: float = 1.0


class EventUnderstandingModel(nn.Module):
    """Tiny event understanding model with dlhub backbone and task-specific heads."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        builder_name = f"build_{cfg.family}_event_model"
        if not hasattr(event_model_zoo, builder_name):
            raise ValueError(f"unknown event model family: {cfg.family}")
        builder = getattr(event_model_zoo, builder_name)
        self.backbone = builder(
            in_channels=int(cfg.in_channels),
            variant=str(cfg.variant),
            width_mult=float(cfg.width_mult),
        )
        self.adapt = nn.LazyConv2d(int(cfg.hidden_channels), kernel_size=1)
        self.body = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(int(cfg.hidden_channels), int(cfg.hidden_channels), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.polarity_head = nn.LazyConv2d(2, kernel_size=1)
        self.motion_head = nn.LazyConv2d(2, kernel_size=1)
        self.depth_head = nn.LazyConv2d(1, kernel_size=1)

    def forward(self, events: torch.Tensor) -> dict[str, torch.Tensor]:
        events = events.to(torch.float32)
        backbone_out = self.backbone(events)
        feat = self.body(self.adapt(backbone_out["event_features"]))
        polarity_map = torch.sigmoid(self.polarity_head(feat))
        motion = torch.tanh(self.motion_head(feat))
        depth_like = torch.sigmoid(self.depth_head(feat))
        return {
            "polarity_map": polarity_map,
            "motion": motion,
            "depth_like": depth_like,
        }


def event_understanding_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    polarity_loss = torch.nn.functional.l1_loss(
        outputs["polarity_map"],
        targets["polarity_map"].to(torch.float32),
    )
    motion_loss = torch.nn.functional.smooth_l1_loss(
        outputs["motion"],
        targets["motion"].to(torch.float32),
    )
    depth_loss = torch.nn.functional.l1_loss(
        outputs["depth_like"],
        targets["depth_like"].to(torch.float32),
    )
    total = polarity_loss + motion_loss + 0.5 * depth_loss
    return total, {
        "polarity_loss": float(polarity_loss.item()),
        "motion_loss": float(motion_loss.item()),
        "depth_loss": float(depth_loss.item()),
    }


def build_model(cfg: ModelConfig) -> EventUnderstandingModel:
    return EventUnderstandingModel(cfg)


__all__ = [
    "ModelConfig",
    "EventUnderstandingModel",
    "build_model",
    "event_understanding_loss",
]
