from __future__ import annotations
import torch
from torch import nn


def check_nchw(x):
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
    return x


class CompactLaneTopology(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int, num_nodes: int):
        super().__init__()
        self.family = str(family)
        self.num_nodes = int(num_nodes)
        c = int(width)
        self.backbone = nn.Sequential(
            nn.Conv2d(int(in_channels), c, 3, 1, 1),
            nn.ReLU(inplace=True),
            *sum(
                [
                    [nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)]
                    for _ in range(max(1, int(depth)))
                ],
                [],
            ),
        )
        self.node = nn.Linear(c, int(num_nodes) * 2)
        self.edge = nn.Linear(c, int(num_nodes) * int(num_nodes))

    def forward(self, image):
        feat = self.backbone(check_nchw(image))
        pooled = feat.mean(dim=(2, 3))
        nodes = self.node(pooled).view(image.shape[0], self.num_nodes, 2)
        edges = self.edge(pooled).view(image.shape[0], self.num_nodes, self.num_nodes)
        return {"nodes": nodes, "edges": edges}


def build_baseline_topology(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    num_nodes: int = 8,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return CompactLaneTopology(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        num_nodes=int(num_nodes),
    )


def smoke_test_topology(builder, variant: str):
    out = builder(in_channels=3, variant=variant, width_mult=0.5, num_nodes=6)(
        torch.randn(2, 3, 128, 128)
    )
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
