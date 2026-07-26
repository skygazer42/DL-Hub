from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def check_nchw(x):
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
    return x


class ToyDepthCompleter(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int):
        super().__init__()
        self.family = str(family)
        c = int(width)
        self.net = nn.Sequential(
            nn.Conv2d(int(in_channels), c, 3, 1, 1),
            nn.ReLU(inplace=True),
            *sum(
                [
                    [nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)]
                    for _ in range(max(1, int(depth)))
                ],
                [],
            ),
            nn.Conv2d(c, 1, 1),
        )

    def forward(self, image):
        depth = F.softplus(self.net(check_nchw(image)))
        return {"depth": depth}


def build_toy_model(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    **kwargs,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return ToyDepthCompleter(
        family=str(family), in_channels=int(in_channels), width=width, depth=int(spec["depth"])
    )


def smoke_test_model(builder, variant: str):
    out = builder(in_channels=4, variant=variant, width_mult=0.5)(torch.randn(2, 4, 64, 64))
    print(variant, tuple(out["depth"].shape))
