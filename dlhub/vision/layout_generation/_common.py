
from __future__ import annotations

import torch
from torch import nn


class ToyVisionDirectionModel(nn.Module):
    def __init__(self, *, width: int, depth: int, in_channels: int):
        super().__init__()
        layers: list[nn.Module] = []
        c = int(in_channels)
        hidden = int(width)
        for _ in range(int(depth)):
            layers.append(nn.Conv2d(c, hidden, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c = hidden
        layers.append(nn.Conv2d(c, int(in_channels), kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_toy_vision_direction(*, family: str, variants: dict[str, dict[str, int]], in_channels: int, variant: str, width_mult: float = 1.0):
    spec = dict(variants[str(variant)])
    width = max(8, int(spec['width'] * float(width_mult)))
    depth = int(spec['depth'])
    return ToyVisionDirectionModel(width=width, depth=depth, in_channels=int(in_channels))


def smoke_test_direction(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(variant, tuple(y.shape))
