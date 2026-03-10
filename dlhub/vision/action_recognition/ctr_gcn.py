"""CTR-GCN (dynamic topology refinement) - toy-first skeleton action classifier.

Reference (idea):
- "Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition"
  (CVPR 2021)

Toy interpretation:
- Replace hand-designed adjacency with a per-frame learned joint-to-joint attention matrix.
- Use attention to aggregate joint features, then temporal conv and residual connections.
"""

import math

import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels

from ._common import check_skeleton_input


class JointAttention(nn.Module):
    def __init__(self, *, channels: int, attn_dim: int) -> None:
        super().__init__()
        c = int(channels)
        d = int(attn_dim)
        if c <= 0 or d <= 0:
            raise ValueError("channels/attn_dim must be > 0")

        self.q = nn.Conv1d(c, d, kernel_size=1, bias=True)
        self.k = nn.Conv1d(c, d, kernel_size=1, bias=True)
        self.v = nn.Conv1d(c, c, kernel_size=1, bias=True)
        self.scale = 1.0 / math.sqrt(float(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        b, c, t, v = x.shape
        x_bt = x.permute(0, 2, 1, 3).reshape(b * t, c, v)  # (BT,C,V)
        q = self.q(x_bt).transpose(1, 2)  # (BT,V,D)
        k = self.k(x_bt)  # (BT,D,V)
        attn = torch.softmax((q @ k) * float(self.scale), dim=-1)  # (BT,V,V)
        val = self.v(x_bt)  # (BT,C,V)
        out = torch.einsum("bcv,bvw->bcw", val, attn)  # (BT,C,V)
        out = out.view(b, t, c, v).permute(0, 2, 1, 3).contiguous()
        return out


class CTRGCNBlock(nn.Module):
    def __init__(self, *, channels: int, kt: int, attn_dim: int) -> None:
        super().__init__()
        c = int(channels)
        k = int(kt)
        if k <= 0:
            raise ValueError("kt must be > 0")
        self.attn = JointAttention(channels=c, attn_dim=int(attn_dim))
        self.tcn = nn.Conv2d(c, c, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attn(x)
        y = self.bn(self.tcn(self.act(y)))
        return self.act(x + y)


class CTRGCNSkeletonClassifier(nn.Module):
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
        del num_joints
        c_in = int(in_channels)
        w = int(width)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Conv2d(c_in, w, kernel_size=1, bias=True)
        attn_dim = max(8, w // 2)
        self.blocks = nn.Sequential(
            *[CTRGCNBlock(channels=w, kt=int(kt), attn_dim=attn_dim) for _ in range(d)]
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
    "ctr_gcn_tiny": {"width": 32, "depth": 2, "kt": 5},
    "ctr_gcn_small": {"width": 48, "depth": 3, "kt": 5},
    "ctr_gcn_base": {"width": 64, "depth": 4, "kt": 9},
}


def build_ctr_gcn_skeleton_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_joints: int = 17,
    seq_len: int = 32,
    variant: str = "ctr_gcn_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    del seq_len
    if int(num_joints) <= 0:
        raise ValueError("num_joints must be > 0")
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CTR-GCN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = scale_channels(int(spec["width"]), float(width_mult), min_ch=16, divisor=8)
    return CTRGCNSkeletonClassifier(
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
    m = build_ctr_gcn_skeleton_classifier(
        in_channels=3,
        num_classes=6,
        num_joints=17,
        seq_len=32,
        variant="ctr_gcn_tiny",
        width_mult=0.5,
        dropout=0.0,
    )
    y = m(x)
    print("ctr_gcn_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
