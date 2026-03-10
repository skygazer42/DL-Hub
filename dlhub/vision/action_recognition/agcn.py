"""2S-AGCN (Adaptive Graph Convolutional Network) - toy-first skeleton action classifier.

Reference (idea):
- "Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition" (CVPR 2019)

Toy interpretation:
- Start from a fixed adjacency (ring + self loops).
- Learn an additive adjacency bias, softmax-normalized per row.
- Use two simple streams (A and A^T) and sum their outputs.
"""

import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels

from ._common import check_skeleton_input
from .stgcn import make_ring_adjacency


class AdaptiveGraphConv(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int, num_joints: int) -> None:
        super().__init__()
        v = int(num_joints)
        self.register_buffer("A_base", make_ring_adjacency(v), persistent=False)
        self.A_bias = nn.Parameter(torch.zeros(v, v))

        self.proj = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=True)

    def _adj(self) -> torch.Tensor:
        a = self.A_base + self.A_bias
        return torch.softmax(a, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        a = self._adj()
        y1 = torch.einsum("bctv,vw->bctw", x, a)
        y2 = torch.einsum("bctv,vw->bctw", x, a.transpose(0, 1))
        y = y1 + y2
        return self.proj(y)


class AGCNBlock(nn.Module):
    def __init__(self, *, channels: int, num_joints: int, kt: int) -> None:
        super().__init__()
        c = int(channels)
        k = int(kt)
        if k <= 0:
            raise ValueError("kt must be > 0")
        self.gcn = AdaptiveGraphConv(in_channels=c, out_channels=c, num_joints=int(num_joints))
        self.tcn = nn.Conv2d(c, c, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gcn(x)
        y = self.bn(self.tcn(self.act(y)))
        return self.act(x + y)


class AGCNSkeletonClassifier(nn.Module):
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
            *[AGCNBlock(channels=w, num_joints=v, kt=int(kt)) for _ in range(d)]
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
    "agcn_tiny": {"width": 32, "depth": 2, "kt": 5},
    "agcn_small": {"width": 48, "depth": 3, "kt": 5},
    "agcn_base": {"width": 64, "depth": 4, "kt": 9},
}


def build_agcn_skeleton_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_joints: int = 17,
    seq_len: int = 32,
    variant: str = "agcn_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    del seq_len
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown AGCN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = scale_channels(int(spec["width"]), float(width_mult), min_ch=16, divisor=8)
    return AGCNSkeletonClassifier(
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
    m = build_agcn_skeleton_classifier(
        in_channels=3,
        num_classes=6,
        num_joints=17,
        seq_len=32,
        variant="agcn_tiny",
        width_mult=0.5,
        dropout=0.0,
    )
    y = m(x)
    print("agcn_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
