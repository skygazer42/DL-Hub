from __future__ import annotations

import torch
from torch import nn


class CompactImageSynthesizer(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int):
        super().__init__()
        self.family = str(family)
        c = int(width)
        layers: list[nn.Module] = [nn.Conv2d(int(in_channels), c, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend([nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)])
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv2d(c, int(in_channels), 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = image.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
        generated = torch.tanh(self.head(self.net(x)))
        return {"generated": generated}


def build_baseline_generator(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return CompactImageSynthesizer(
        family=str(family), in_channels=int(in_channels), width=width, depth=int(spec["depth"])
    )


def smoke_test_generator(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 64, 64))
    print(variant, tuple(out["generated"].shape))
