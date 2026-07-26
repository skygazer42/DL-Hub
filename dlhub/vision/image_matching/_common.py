from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


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


class ToyImageMatcher(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int, embed_dim: int):
        super().__init__()
        self.family = str(family)
        self.enc = TinyEncoder(in_channels, width, depth)
        self.proj = nn.Linear(self.enc.out_channels, int(embed_dim))

    def forward(self, image1, image2):
        f1 = self.enc(image1).mean(dim=(2, 3))
        f2 = self.enc(image2).mean(dim=(2, 3))
        e1 = F.normalize(self.proj(f1), dim=1)
        e2 = F.normalize(self.proj(f2), dim=1)
        return {"similarity": (e1 * e2).sum(dim=1), "embedding1": e1, "embedding2": e2}


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
    embed = max(64, int(int(spec["embed"]) * float(width_mult)))
    return ToyImageMatcher(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        embed_dim=embed,
    )


def smoke_test_model(builder, variant: str):
    out = builder(in_channels=3, variant=variant, width_mult=0.5)(
        torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    )
    print(
        variant,
        {k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__ for k, v in out.items()},
    )
