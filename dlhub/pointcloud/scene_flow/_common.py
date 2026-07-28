from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def _check_pair(
    points1: torch.Tensor,
    points2: torch.Tensor,
    *,
    in_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    src = points1.to(torch.float32)
    tgt = points2.to(torch.float32)
    if src.ndim != 3 or tgt.ndim != 3:
        raise ValueError(
            f"Expected scene-flow inputs with shape (B, N, C), got {tuple(src.shape)} and {tuple(tgt.shape)}"
        )
    if src.shape != tgt.shape:
        raise ValueError(
            f"Source/target point clouds must match shape, got {tuple(src.shape)} vs {tuple(tgt.shape)}"
        )
    if src.shape[-1] != int(in_channels):
        raise ValueError(f"Expected {int(in_channels)} channels, got {int(src.shape[-1])}")
    return src, tgt


def _xyz(points: torch.Tensor) -> torch.Tensor:
    if points.shape[-1] >= 3:
        return points[..., :3]
    pad = points.new_zeros(*points.shape[:-1], 3 - points.shape[-1])
    return torch.cat([points, pad], dim=-1)


class _ResidualMixer(nn.Module):
    def __init__(self, *, width: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        hidden = max(width, int(width) * int(hidden_mult))
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        ctx = context.expand(-1, x.shape[1], -1)
        return x + self.ffn(torch.cat([self.norm(x), ctx], dim=-1))


class CompactSceneFlowEstimator(nn.Module):
    """Compact scene-flow estimator for local zoo coverage.

    The model is intentionally lightweight: it encodes paired point features, mixes
    them with global scene context, and predicts per-point XYZ motion.
    """

    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        width: int,
        depth: int,
        hidden_mult: int,
        refine_steps: int,
        delta_scale: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.in_channels = int(in_channels)
        self.delta_scale = float(delta_scale)
        pair_width = int(self.in_channels) * 2 + 3
        self.input_proj = nn.Sequential(
            nn.Linear(pair_width, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(pair_width, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.blocks = nn.ModuleList(
            [
                _ResidualMixer(width=width, hidden_mult=hidden_mult, dropout=dropout)
                for _ in range(max(1, int(depth)))
            ]
        )
        self.output_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 3)
        self.refiners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width + 3, width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(width, 3),
                )
                for _ in range(max(0, int(refine_steps) - 1))
            ]
        )

    def forward(self, points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        src, tgt = _check_pair(points1, points2, in_channels=self.in_channels)
        delta = _xyz(tgt) - _xyz(src)
        pair = torch.cat([src, tgt, delta], dim=-1)
        x = self.input_proj(pair)
        context = self.context_proj(pair.mean(dim=1, keepdim=True))
        for block in self.blocks:
            x = block(x, context)
        x = self.output_norm(x)
        flow = self.head(x) + self.delta_scale * delta
        for refiner in self.refiners:
            flow = flow + 0.1 * refiner(torch.cat([x, flow], dim=-1))
        return flow


def build_scene_flow_estimator(
    *,
    family: str,
    in_channels: int,
    variant: str,
    variants: dict[str, dict[str, int | float]],
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = variants[str(variant)]
    width = max(16, int(int(cfg["width"]) * float(width_mult)))
    return CompactSceneFlowEstimator(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        hidden_mult=int(cfg.get("hidden_mult", 2)),
        refine_steps=int(cfg.get("refine_steps", 1)),
        delta_scale=float(cfg.get("delta_scale", 1.0)),
        dropout=float(dropout),
    )


def smoke_test_scene_flow_estimator(
    builder: Callable[..., nn.Module],
    variant: str,
) -> None:
    torch.manual_seed(0)
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    src = torch.randn(2, 64, 3)
    tgt = src + 0.1 * torch.randn(2, 64, 3)
    flow = model(src, tgt)
    flow.mean().backward()
    print(variant, tuple(flow.shape))
