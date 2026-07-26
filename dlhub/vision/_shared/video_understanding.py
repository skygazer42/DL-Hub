from __future__ import annotations

import torch
from torch import nn


def check_btchw(x):
    x = x.to(torch.float32)
    if x.ndim != 5:
        raise ValueError(f"Expected input shape (B,T,C,H,W), got {tuple(x.shape)}")
    return x


class ToyVideoUnderstander(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int, num_classes: int):
        super().__init__()
        self.family = str(family)
        c = int(width)
        self.frame = nn.Sequential(nn.Conv2d(int(in_channels), c, 3, 1, 1), nn.ReLU(inplace=True))
        self.temporal = nn.GRU(c, c, batch_first=True)
        self.head = nn.Linear(c, int(num_classes))

    def forward(self, video):
        x = check_btchw(video)
        b, t, c, h, w = x.shape
        feat = self.frame(x.view(b * t, c, h, w)).mean(dim=(2, 3)).view(b, t, -1)
        seq, _ = self.temporal(feat)
        logits = self.head(seq[:, -1])
        return {"logits": logits}


def build_toy_video_understander(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return ToyVideoUnderstander(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        num_classes=int(num_classes),
    )


def smoke_test_vu(builder, variant: str):
    out = builder(in_channels=3, num_classes=6, variant=variant, width_mult=0.5)(
        torch.randn(2, 4, 3, 64, 64)
    )
    print(variant, tuple(out["logits"].shape))
