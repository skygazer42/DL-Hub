"""Shift-GCN - toy-first skeleton action classifier.

Reference (idea):
- "Shift-GCN: Shift Graph Convolutional Network for Skeleton-based Action Recognition" (AAAI 2020)

Toy interpretation:
- Replace expensive graph ops with cheap "shift" operations along time/joint dimensions.
- Follow with pointwise (1x1) channel mixing and a temporal conv.
"""

import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels

from ._common import check_skeleton_input


def _shift_along_time(x: torch.Tensor, *, fold_div: int = 8) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected skeleton input shape (B, C, T, V), got {tuple(x.shape)}")
    b, c, t, v = x.shape
    if t <= 1:
        return x
    d = max(1, int(fold_div))
    fold = max(1, c // d)
    out = x.clone()
    out[:, :fold, 1:] = x[:, :fold, :-1]
    out[:, fold : 2 * fold, :-1] = x[:, fold : 2 * fold, 1:]
    return out


def _shift_along_joints(x: torch.Tensor, *, fold_div: int = 8) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected skeleton input shape (B, C, T, V), got {tuple(x.shape)}")
    b, c, t, v = x.shape
    if v <= 1:
        return x
    d = max(1, int(fold_div))
    fold = max(1, c // d)
    out = x.clone()
    out[:, :fold, :, 1:] = x[:, :fold, :, :-1]
    out[:, fold : 2 * fold, :, :-1] = x[:, fold : 2 * fold, :, 1:]
    return out


class ShiftGCNBlock(nn.Module):
    def __init__(self, *, channels: int, kt: int, fold_div: int) -> None:
        super().__init__()
        c = int(channels)
        k = int(kt)
        if c <= 0:
            raise ValueError("channels must be > 0")
        if k <= 0:
            raise ValueError("kt must be > 0")
        self.fold_div = int(fold_div)

        self.mix = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.tcn = nn.Conv2d(c, c, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = _shift_along_time(x, fold_div=int(self.fold_div))
        y = _shift_along_joints(y, fold_div=int(self.fold_div))
        y = self.mix(y)
        y = self.bn(self.tcn(self.act(y)))
        return self.act(identity + y)


class ShiftGCNSkeletonClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        num_joints: int,
        width: int,
        depth: int,
        kt: int,
        fold_div: int,
        dropout: float,
    ) -> None:
        super().__init__()
        del num_joints
        c_in = int(in_channels)
        w = int(width)
        d = int(depth)
        if w < 8:
            raise ValueError("width must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Conv2d(c_in, w, kernel_size=1, bias=True)
        self.blocks = nn.Sequential(
            *[ShiftGCNBlock(channels=w, kt=int(kt), fold_div=int(fold_div)) for _ in range(d)]
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
    "shift_gcn_tiny": {"width": 32, "depth": 2, "kt": 5, "fold_div": 8},
    "shift_gcn_small": {"width": 48, "depth": 3, "kt": 5, "fold_div": 8},
    "shift_gcn_base": {"width": 64, "depth": 4, "kt": 9, "fold_div": 8},
}


def build_shift_gcn_skeleton_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_joints: int = 17,
    seq_len: int = 32,
    variant: str = "shift_gcn_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    del seq_len
    if int(num_joints) <= 0:
        raise ValueError("num_joints must be > 0")
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Shift-GCN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = scale_channels(int(spec["width"]), float(width_mult), min_ch=16, divisor=8)
    return ShiftGCNSkeletonClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        num_joints=int(num_joints),
        width=int(width),
        depth=int(spec["depth"]),
        kt=int(spec["kt"]),
        fold_div=int(spec["fold_div"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 17)
    m = build_shift_gcn_skeleton_classifier(
        in_channels=3,
        num_classes=6,
        num_joints=17,
        seq_len=32,
        variant="shift_gcn_tiny",
        width_mult=0.5,
        dropout=0.0,
    )
    y = m(x)
    print("shift_gcn_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
