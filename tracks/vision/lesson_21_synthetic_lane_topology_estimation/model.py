from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 24
    num_blocks: int = 3
    max_lanes: int = 4


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


class LaneTopologyEstimator(nn.Module):
    """Predict lane-specific heatmaps and a lane-lane adjacency matrix."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        widths = [int(cfg.hidden_channels) * (2**i) for i in range(int(cfg.num_blocks))]
        self.max_lanes = int(cfg.max_lanes)

        self.stem = ConvBlock(int(cfg.in_channels), widths[0])
        self.encoder = nn.ModuleList(
            [ConvBlock(widths[i], widths[min(i + 1, len(widths) - 1)]) for i in range(len(widths) - 1)]
        )
        self.decoder = nn.ModuleList(
            [ConvBlock(widths[i + 1] + widths[i], widths[i]) for i in range(len(widths) - 2, -1, -1)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.heatmap_head = nn.Conv2d(widths[0], self.max_lanes, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lane_queries = nn.Parameter(torch.randn(self.max_lanes, widths[0]) * 0.02)
        self.graph_proj = nn.Linear(widths[0], widths[0], bias=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        skips: list[torch.Tensor] = []

        feat = self.stem(x)
        skips.append(feat)
        for block in self.encoder:
            feat = self.pool(feat)
            feat = block(feat)
            skips.append(feat)

        feat = skips.pop()
        for block in self.decoder:
            skip = skips.pop()
            feat = torch.nn.functional.interpolate(
                feat,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            feat = block(torch.cat([feat, skip], dim=1))

        lane_heatmaps = torch.sigmoid(self.heatmap_head(feat))
        pooled = self.global_pool(feat).flatten(1)
        node_repr = pooled.unsqueeze(1) + self.lane_queries.unsqueeze(0)
        node_repr = self.graph_proj(node_repr)
        adjacency_logits = torch.matmul(node_repr, node_repr.transpose(1, 2)) / (
            node_repr.shape[-1] ** 0.5
        )
        return {
            "lane_heatmaps": lane_heatmaps,
            "adjacency_logits": adjacency_logits,
        }


def lane_topology_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    heatmap_loss = torch.nn.functional.mse_loss(
        outputs["lane_heatmaps"],
        targets["lane_heatmaps"],
    )

    adjacency_target = targets["adjacency"].to(torch.float32)
    lane_presence = targets["lane_presence"].to(torch.float32)
    pair_mask = torch.einsum("bi,bj->bij", lane_presence, lane_presence)
    off_diag = 1.0 - torch.eye(pair_mask.shape[-1], device=pair_mask.device).unsqueeze(0)
    pair_mask = pair_mask * off_diag

    adjacency_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["adjacency_logits"],
        adjacency_target,
        reduction="none",
    )
    adjacency_loss = (adjacency_loss * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

    total_loss = heatmap_loss + adjacency_loss
    parts = {
        "heatmap_loss": float(heatmap_loss.item()),
        "adjacency_loss": float(adjacency_loss.item()),
    }
    return total_loss, parts


def build_model(cfg: ModelConfig) -> LaneTopologyEstimator:
    return LaneTopologyEstimator(cfg)


__all__ = [
    "LaneTopologyEstimator",
    "ModelConfig",
    "build_model",
    "lane_topology_loss",
]
