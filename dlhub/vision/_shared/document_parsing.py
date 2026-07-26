from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(x):
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
    return x


class TinyEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int, depth: int):
        super().__init__()
        c = int(width)
        layers = [nn.Conv2d(int(in_channels), c, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(max(1, int(depth))):
            layers += [nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        return self.net(check_nchw(x))


class ToyModel(nn.Module):
    def __init__(
        self, *, family: str, in_channels: int, width: int, depth: int, num_regions: int = 8
    ):
        super().__init__()
        self.family = str(family)
        self.enc = TinyEncoder(in_channels, width, depth)
        c = self.enc.out_channels
        self.cls = nn.Linear(c, int(num_regions))
        self.box = nn.Linear(c, 4)

    def forward(self, image):
        feat = self.enc(image)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return {"region_logits": self.cls(pooled), "boxes": torch.sigmoid(self.box(pooled))}


def build_toy_model(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    num_regions: int = 8,
    **kwargs,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return ToyModel(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        num_regions=int(num_regions),
    )


def smoke_test_model(builder, variant: str):
    out = builder(in_channels=3, variant=variant, width_mult=0.5, num_regions=8)(
        torch.randn(2, 3, 128, 128)
    )
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
