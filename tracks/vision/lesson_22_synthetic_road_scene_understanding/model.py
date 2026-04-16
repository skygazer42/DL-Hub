from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 24
    num_blocks: int = 3
    num_lane_slots: int = 3
    num_object_types: int = 3
    num_scene_classes: int = 4


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RoadSceneUnderstandingModel(nn.Module):
    """Predict lane availability, object presence, and a fused scene class."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        widths = [int(cfg.hidden_channels) * (2**i) for i in range(int(cfg.num_blocks))]
        self.num_lane_slots = int(cfg.num_lane_slots)
        self.num_object_types = int(cfg.num_object_types)

        self.stem = ConvBlock(int(cfg.in_channels), widths[0])
        self.encoder = nn.ModuleList(
            [ConvBlock(widths[i], widths[min(i + 1, len(widths) - 1)]) for i in range(len(widths) - 1)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.scene_proj = nn.Linear(widths[-1], widths[0])
        self.lane_queries = nn.Parameter(torch.randn(self.num_lane_slots, widths[0]) * 0.02)
        self.object_queries = nn.Parameter(torch.randn(self.num_object_types, widths[0]) * 0.02)
        self.lane_head = nn.Linear(widths[0], 1)
        self.object_head = nn.Linear(widths[0], 1)
        self.scene_head = nn.Linear(widths[0], int(cfg.num_scene_classes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        feat = self.stem(x)
        for block in self.encoder:
            feat = self.pool(feat)
            feat = block(feat)

        pooled = self.global_pool(feat).flatten(1)
        scene_token = self.scene_proj(pooled)
        lane_tokens = scene_token.unsqueeze(1) + self.lane_queries.unsqueeze(0)
        object_tokens = scene_token.unsqueeze(1) + self.object_queries.unsqueeze(0)
        fused_scene = scene_token + lane_tokens.mean(dim=1) + object_tokens.mean(dim=1)

        lane_logits = self.lane_head(lane_tokens).squeeze(-1)
        object_logits = self.object_head(object_tokens).squeeze(-1)
        scene_logits = self.scene_head(fused_scene)
        return {
            "lane_logits": lane_logits,
            "object_logits": object_logits,
            "scene_logits": scene_logits,
        }


def road_scene_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    lane_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["lane_logits"],
        targets["lane_targets"].to(torch.float32),
    )
    object_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["object_logits"],
        targets["object_targets"].to(torch.float32),
    )
    scene_loss = torch.nn.functional.cross_entropy(
        outputs["scene_logits"],
        targets["scene_label"].to(torch.long),
    )
    total_loss = lane_loss + object_loss + scene_loss
    parts = {
        "lane_loss": float(lane_loss.item()),
        "object_loss": float(object_loss.item()),
        "scene_loss": float(scene_loss.item()),
    }
    return total_loss, parts


def build_model(cfg: ModelConfig) -> RoadSceneUnderstandingModel:
    return RoadSceneUnderstandingModel(cfg)


__all__ = [
    "ModelConfig",
    "RoadSceneUnderstandingModel",
    "build_model",
    "road_scene_loss",
]
