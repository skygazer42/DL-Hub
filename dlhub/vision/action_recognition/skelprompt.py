"""ST-GCN - toy-first skeleton action classifier.

Reference:
- "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition" (AAAI 2018)

Toy interpretation:
- Use a fixed adjacency (ring + self loops) and simple graph aggregation.
- Apply temporal conv and global average pooling for classification.
"""

import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels

from ._common import check_skeleton_input


def make_ring_adjacency(num_joints: int) -> torch.Tensor:
    v = int(num_joints)
    if v <= 0:
        raise ValueError("num_joints must be > 0")

    a = torch.eye(v, dtype=torch.float32)
    for i in range(v - 1):
        a[i, i + 1] = 1.0
        a[i + 1, i] = 1.0
    if v > 2:
        a[0, v - 1] = 1.0
        a[v - 1, 0] = 1.0

    # Row-normalize so aggregation is stable.
    deg = a.sum(dim=1, keepdim=True).clamp_min(1.0)
    return a / deg


class GraphConv(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int, num_joints: int) -> None:
        super().__init__()
        self.register_buffer("A", make_ring_adjacency(int(num_joints)), persistent=False)
        self.proj = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, T, V), got {tuple(x.shape)}")
        x = torch.einsum("bctv,vw->bctw", x, self.A)  # aggregate neighbors
        return self.proj(x)


class STGCNBlock(nn.Module):
    def __init__(self, *, channels: int, num_joints: int, kt: int) -> None:
        super().__init__()
        c = int(channels)
        k = int(kt)
        if k <= 0:
            raise ValueError("kt must be > 0")

        self.gcn = GraphConv(in_channels=c, out_channels=c, num_joints=int(num_joints))
        self.tcn = nn.Conv2d(c, c, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gcn(x)
        y = self.bn(self.tcn(self.act(y)))
        return self.act(x + y)


class SkelpromptSkeletonClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_joints: int,
        width: int,
        depth: int,
        kt: int,
        dropout: float,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        v = int(num_joints)
        w = int(width)
        d = int(depth)
        if v <= 0:
            raise ValueError("num_joints must be > 0")
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Conv2d(c_in, w, kernel_size=1, bias=True)
        self.blocks = nn.Sequential(
            *[STGCNBlock(channels=w, num_joints=v, kt=int(kt)) for _ in range(d)]
        )
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(w, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_skeleton_input(x)  # (B,C,T,V)
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))  # global avg over (T,V)
        x = self.dropout(x)
        return self.classifier(x)


_VARIANTS: dict[str, dict] = {
    "skelprompt_tiny": {"width": 32, "depth": 2, "kt": 5},
    "skelprompt_small": {"width": 48, "depth": 3, "kt": 5},
    "skelprompt_base": {"width": 64, "depth": 4, "kt": 9},
}


def build_skelprompt_skeleton_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_joints: int = 17,
    seq_len: int = 32,
    variant: str = "skelprompt_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    del seq_len
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ST-GCN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = scale_channels(int(spec["width"]), float(width_mult), min_ch=16, divisor=8)
    return SkelpromptSkeletonClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        num_joints=int(num_joints),
        width=int(width),
        depth=int(spec["depth"]),
        kt=int(spec["kt"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 17)
    m = build_skelprompt_skeleton_classifier(
        in_channels=3,
        num_classes=6,
        num_joints=17,
        seq_len=32,
        variant="skelprompt_tiny",
        width_mult=0.5,
        dropout=0.0,
    )
    y = m(x)
    print("skelprompt_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
