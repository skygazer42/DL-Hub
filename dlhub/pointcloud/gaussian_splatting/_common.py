from __future__ import annotations

import torch
from torch import nn


class ToyGaussianSplatModel(nn.Module):
    def __init__(self, *, width: int, depth: int, in_channels: int):
        super().__init__()
        self.depth = int(depth)
        self.encoder = nn.Sequential(
            nn.Linear(int(in_channels), int(width)),
            nn.ReLU(inplace=True),
            nn.Linear(int(width), int(width)),
            nn.ReLU(inplace=True),
        )
        self.mean_head = nn.Linear(int(width), 3)
        self.scale_head = nn.Linear(int(width), 1)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(points)
        return {
            "means": self.mean_head(feat),
            "scales": self.scale_head(feat).sigmoid(),
            "depth_hint": torch.full(
                (points.shape[0], points.shape[1], 1), float(self.depth), device=points.device
            ),
        }


def build_toy_splatter(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
):
    spec = dict(variants[str(variant)])
    width = max(16, int(spec["width"] * float(width_mult)))
    depth = int(spec["depth"])
    return ToyGaussianSplatModel(width=width, depth=depth, in_channels=int(in_channels))


def smoke_test_splatter(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant)
    x = torch.randn(2, 64, 3)
    y = model(x)
    print(variant, sorted(y.keys()))
