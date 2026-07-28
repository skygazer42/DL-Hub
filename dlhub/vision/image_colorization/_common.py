from __future__ import annotations
import torch
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


class CompactColorizer(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int):
        super().__init__()
        self.family = str(family)
        self.enc = TinyEncoder(in_channels, width, depth)
        self.head = nn.Conv2d(self.enc.out_channels, 2, 1)

    def forward(self, image):
        gray = check_nchw(image)[:, :1]
        ab = torch.tanh(self.head(self.enc(gray)))
        return {"ab": ab}


def build_baseline_model(
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
    return CompactColorizer(family=str(family), in_channels=1, width=width, depth=int(spec["depth"]))


def smoke_test_model(builder, variant: str):
    out = builder(in_channels=1, variant=variant, width_mult=0.5)(torch.rand(2, 1, 64, 64))
    print(variant, tuple(out["ab"].shape))
