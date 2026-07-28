from __future__ import annotations

import torch
from torch import nn


class CompactEventModel(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int, steps: int):
        super().__init__()
        self.family = str(family)
        c = int(width)
        layers: list[nn.Module] = [nn.Conv2d(int(in_channels), c, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend([nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)])
        self.encoder = nn.Sequential(*layers)
        self.refiner = nn.Conv2d(c, c, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, 6)
        self.steps = max(1, int(steps))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = image.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
        feat = self.encoder(x)
        for _ in range(self.steps):
            feat = feat + torch.tanh(self.refiner(feat))
        pooled = self.pool(feat).flatten(1)
        return {"logits": self.head(pooled), "event_features": feat, "pooled": pooled}


def build_baseline_event_model(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return CompactEventModel(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        steps=int(spec["steps"]),
    )


def smoke_test_event_model(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 40, 40))
    print(variant, tuple(out["logits"].shape), tuple(out["event_features"].shape))
