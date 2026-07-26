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
        return F.adaptive_avg_pool2d(self.net(check_nchw(x)), (1, 1)).flatten(1)


class ToyMeshRecovery(nn.Module):
    def __init__(
        self, *, family: str, in_channels: int, width: int, depth: int, num_vertices: int = 32
    ):
        super().__init__()
        self.family = str(family)
        self.num_vertices = int(num_vertices)
        self.enc = TinyEncoder(in_channels, width, depth)
        self.shape = nn.Linear(self.enc.out_channels, int(num_vertices) * 3)
        self.pose = nn.Linear(self.enc.out_channels, 72)

    def forward(self, image):
        feat = self.enc(image)
        verts = self.shape(feat).view(image.shape[0], self.num_vertices, 3)
        pose = self.pose(feat)
        return {"vertices": verts, "pose": pose}


def build_toy_mesh(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    num_vertices: int = 32,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return ToyMeshRecovery(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        num_vertices=int(num_vertices),
    )


def smoke_test_mesh(builder, variant: str):
    out = builder(in_channels=3, variant=variant, width_mult=0.5, num_vertices=32)(
        torch.randn(2, 3, 128, 128)
    )
    print(variant, tuple(out["vertices"].shape), tuple(out["pose"].shape))
