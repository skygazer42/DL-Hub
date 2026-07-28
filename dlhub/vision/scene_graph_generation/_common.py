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


class CompactSceneGraph(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        num_objects: int,
        num_relations: int,
        width: int,
        depth: int,
    ):
        super().__init__()
        self.family = str(family)
        self.enc = TinyEncoder(in_channels, width, depth)
        c = self.enc.out_channels
        self.obj = nn.Linear(c, int(num_objects))
        self.rel = nn.Linear(c, int(num_relations))

    def forward(self, image):
        feat = self.enc(image)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return {"object_logits": self.obj(pooled), "relation_logits": self.rel(pooled)}


def build_baseline_model(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    num_objects: int,
    num_relations: int,
    variant: str,
    width_mult: float = 1.0,
    **kwargs,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return CompactSceneGraph(
        family=str(family),
        in_channels=int(in_channels),
        num_objects=int(num_objects),
        num_relations=int(num_relations),
        width=width,
        depth=int(spec["depth"]),
    )


def smoke_test_model(builder, variant: str):
    out = builder(in_channels=3, num_objects=20, num_relations=15, variant=variant, width_mult=0.5)(
        torch.randn(2, 3, 64, 64)
    )
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
