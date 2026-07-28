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
        layers = [nn.Conv2d(int(in_channels), c, 3, 2, 1), nn.ReLU(inplace=True)]
        for _ in range(max(1, int(depth))):
            layers += [nn.Conv2d(c, c, 3, 2, 1), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        feat = self.net(check_nchw(x))
        return F.normalize(F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1), dim=1)


class CompactFewShot(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int, embed_dim: int):
        super().__init__()
        self.family = str(family)
        self.enc = TinyEncoder(in_channels, width, depth)
        self.proj = nn.Linear(self.enc.out_channels, int(embed_dim))

    def forward(self, support, query):
        s = F.normalize(self.proj(self.enc(support)), dim=1)
        q = F.normalize(self.proj(self.enc(query)), dim=1)
        return {"similarity": q @ s.t()}


def build_baseline_fewshot(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    embed = max(64, int(int(spec["embed"]) * float(width_mult)))
    return CompactFewShot(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        embed_dim=embed,
    )


def smoke_test_few(builder, variant: str):
    out = builder(in_channels=3, variant=variant, width_mult=0.5)(
        torch.randn(5, 3, 64, 64), torch.randn(2, 3, 64, 64)
    )
    print(variant, tuple(out["similarity"].shape))
