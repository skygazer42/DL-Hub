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


class ToyModel(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int):
        super().__init__()
        self.family = str(family)
        self.enc = TinyEncoder(in_channels + 1, width, depth)
        self.head = nn.Conv2d(self.enc.out_channels, int(in_channels), 3, 1, 1)

    def forward(self, image, mask=None):
        x = check_nchw(image)
        m = (
            torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            if mask is None
            else mask.to(x.dtype)
        )
        feat = self.enc(torch.cat([x, m], dim=1))
        filled = torch.tanh(self.head(feat))
        out = x * (1 - m) + filled * m
        return {"inpainted": out, "mask": m}


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
    return ToyModel(
        family=str(family), in_channels=int(in_channels), width=width, depth=int(spec["depth"])
    )


def smoke_test_model(builder, variant: str):
    x = torch.randn(2, 3, 64, 64)
    m = torch.zeros(2, 1, 64, 64)
    m[:, :, 16:48, 16:48] = 1
    out = builder(in_channels=3, variant=variant, width_mult=0.5)(x, m)
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
