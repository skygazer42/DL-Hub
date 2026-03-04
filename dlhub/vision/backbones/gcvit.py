from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath, GlobalAvgPoolHead, LayerNorm2d, scale_channels
from dlhub.vision.backbones._transformer import TransformerEncoderBlock


class GlobalContextAdd(nn.Module):
    """GCViT-like global context: add a learned transform of global pooled feature."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        d = int(dim)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(d, d, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class GCViTBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.gc = GlobalContextAdd(d)
        self.dp0 = DropPath(float(drop_path))
        self.attn = TransformerEncoderBlock(d, int(num_heads), mlp_ratio=4.0, dropout=0.0, drop_path=float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp0(self.gc(x))
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = self.attn(t)
        return t.transpose(1, 2).contiguous().view(b, c, h, w)


class GCViTClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)
        depths = tuple(int(x) for x in depths)
        heads = tuple(int(h) for h in heads)
        total = sum(depths)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=total).tolist()
        dp_iter = iter(dp_rates)

        self.down = nn.ModuleList()
        self.down.append(nn.Sequential(nn.Conv2d(int(in_channels), dims[0], kernel_size=4, stride=4), LayerNorm2d(dims[0])))
        for i in range(3):
            self.down.append(nn.Sequential(LayerNorm2d(dims[i]), nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)))

        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[GCViTBlock(dims[i], num_heads=heads[i], drop_path=float(next(dp_iter))) for _ in range(depths[i])]))
        self.head = GlobalAvgPoolHead(dims[-1], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        for i in range(4):
            x = self.down[i](x)
            x = self.stages[i](x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "gcvit_tiny": {"dims": (96, 192, 384, 768), "depths": (2, 2, 6, 2), "heads": (3, 6, 12, 24)},
    "gcvit_small": {"dims": (96, 192, 384, 768), "depths": (3, 3, 9, 3), "heads": (3, 6, 12, 24)},
}


def build_gcvit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "gcvit_tiny",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown GCViT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return GCViTClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        heads=tuple(map(int, spec["heads"])),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_gcvit_classifier(in_channels=3, num_classes=10, variant="gcvit_tiny", width_mult=0.5)
    y = m(x)
    print("gcvit_tiny", tuple(y.shape))

