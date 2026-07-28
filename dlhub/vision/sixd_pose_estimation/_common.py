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
            layers += [nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        return self.net(check_nchw(x))


class CompactPose6D(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int, num_objects: int):
        super().__init__()
        self.family = str(family)
        self.enc = TinyEncoder(in_channels, width, depth)
        c = self.enc.out_channels
        self.rot = nn.Linear(c, 6)
        self.trans = nn.Linear(c, 3)
        self.obj = nn.Linear(c, int(num_objects))

    def forward(self, image):
        feat = self.enc(image)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return {
            "rotation6d": self.rot(pooled),
            "translation": self.trans(pooled),
            "object_logits": self.obj(pooled),
        }


def build_baseline_pose6d(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    num_objects: int,
    variant: str,
    width_mult: float = 1.0,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return CompactPose6D(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        num_objects=int(num_objects),
    )


def smoke_test_6d(builder, variant: str):
    model = builder(in_channels=3, num_objects=8, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 128, 128))
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
