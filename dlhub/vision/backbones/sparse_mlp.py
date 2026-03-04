from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import GlobalAvgPoolHead, LayerNorm2d, scale_channels


class SparseTokenMix(nn.Module):
    """SparseMLP-like sparse mixing via dilated depthwise conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.dw1 = nn.Conv2d(c, c, kernel_size=3, padding=1, dilation=1, groups=c, bias=False)
        self.dw2 = nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw1(x) + self.dw2(x)
        return self.pw(y)


class SparseMLPBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = LayerNorm2d(d)
        self.mix = SparseTokenMix(d)
        self.norm2 = LayerNorm2d(d)
        self.mlp = nn.Sequential(nn.Conv2d(d, 4 * d, kernel_size=1), nn.GELU(), nn.Conv2d(4 * d, d, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SparseMLPClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dim: int = 192,
        depth: int = 10,
        patch_size: int = 4,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d = scale_channels(int(dim), float(width_mult), min_ch=16, divisor=8)
        p = int(patch_size)
        self.patch = nn.Sequential(nn.Conv2d(int(in_channels), d, kernel_size=p, stride=p), LayerNorm2d(d))
        self.blocks = nn.Sequential(*[SparseMLPBlock(d) for _ in range(int(depth))])
        self.head = GlobalAvgPoolHead(d, int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x)
        x = self.blocks(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "sparse_mlp_tiny": {"dim": 192, "depth": 10, "patch": 4},
    "sparse_mlp_small": {"dim": 256, "depth": 12, "patch": 4},
}


def build_sparse_mlp_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "sparse_mlp_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SparseMLP variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return SparseMLPClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        patch_size=int(spec["patch"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_sparse_mlp_classifier(in_channels=3, num_classes=10, variant="sparse_mlp_tiny", width_mult=0.5)
    y = m(x)
    print("sparse_mlp_tiny", tuple(y.shape))

