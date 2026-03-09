"""MS-G3D (Multi-Scale Graph 3D) - toy-first skeleton action classifier.

Reference (idea):
- "MS-G3D: Multi-Scale Graph 3D Convolution Network for Skeleton-based Action Recognition" (CVPR 2020)

Toy interpretation:
- Build multiple k-hop adjacency matrices (ring graph) and aggregate them.
- Follow with a temporal conv + residual connections, then global average pool.
"""

import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels

from ._common import check_skeleton_input
from .stgcn import make_ring_adjacency


def _make_k_hop(a: torch.Tensor, k: int) -> torch.Tensor:
    kk = int(k)
    if kk <= 0:
        raise ValueError("k must be > 0")
    # Treat any positive entry as an edge, then take matrix power.
    base = (a > 0).to(torch.float32)
    hop = torch.matrix_power(base, kk)
    hop = (hop > 0).to(torch.float32)
    deg = hop.sum(dim=1, keepdim=True).clamp_min(1.0)
    return hop / deg


class MultiScaleGraphConv(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int, num_joints: int, hops: int = 2) -> None:
        super().__init__()
        v = int(num_joints)
        if v <= 0:
            raise ValueError("num_joints must be > 0")
        h = int(hops)
        if h <= 0:
            raise ValueError("hops must be > 0")

        a1 = make_ring_adjacency(v)
        adjs = [_make_k_hop(a1, k=i) for i in range(1, h + 1)]
        self.register_buffer("A", torch.stack(adjs, dim=0), persistent=False)  # (K, V, V)
        self.proj = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, T, V), got {tuple(x.shape)}")
        # Sum K-hop aggregations.
        y = 0.0
        for k in range(int(self.A.shape[0])):
            y = y + torch.einsum("bctv,vw->bctw", x, self.A[k])
        return self.proj(y)


class MSG3DBlock(nn.Module):
    def __init__(self, *, channels: int, num_joints: int, kt: int, hops: int) -> None:
        super().__init__()
        c = int(channels)
        k = int(kt)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if k <= 0:
            raise ValueError("kt must be > 0")

        self.gcn = MultiScaleGraphConv(in_channels=c, out_channels=c, num_joints=int(num_joints), hops=int(hops))
        self.tcn = nn.Conv2d(c, c, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gcn(x)
        y = self.bn(self.tcn(self.act(y)))
        return self.act(x + y)


class MSG3DSkeletonClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_joints: int,
        width: int,
        depth: int,
        kt: int,
        hops: int,
        dropout: float,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        v = int(num_joints)
        w = int(width)
        d = int(depth)
        if v <= 0:
            raise ValueError("num_joints must be > 0")
        if w < 8:
            raise ValueError("width must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Conv2d(c_in, w, kernel_size=1, bias=True)
        self.blocks = nn.Sequential(
            *[MSG3DBlock(channels=w, num_joints=v, kt=int(kt), hops=int(hops)) for _ in range(d)]
        )
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(w, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_skeleton_input(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))
        x = self.dropout(x)
        return self.classifier(x)


_VARIANTS: dict[str, dict] = {
    "ms_g3d_tiny": {"width": 32, "depth": 2, "kt": 5, "hops": 2},
    "ms_g3d_small": {"width": 48, "depth": 3, "kt": 5, "hops": 2},
    "ms_g3d_base": {"width": 64, "depth": 4, "kt": 9, "hops": 3},
}


def build_ms_g3d_skeleton_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_joints: int = 17,
    seq_len: int = 32,
    variant: str = "ms_g3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    del seq_len
    if int(num_joints) <= 0:
        raise ValueError("num_joints must be > 0")
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MS-G3D variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = scale_channels(int(spec["width"]), float(width_mult), min_ch=16, divisor=8)
    return MSG3DSkeletonClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        num_joints=int(num_joints),
        width=int(width),
        depth=int(spec["depth"]),
        kt=int(spec["kt"]),
        hops=int(spec["hops"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 17)
    m = build_ms_g3d_skeleton_classifier(in_channels=3, num_classes=6, num_joints=17, seq_len=32, variant="ms_g3d_tiny", width_mult=0.5, dropout=0.0)
    y = m(x)
    print("ms_g3d_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

