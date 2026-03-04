from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath, GlobalAvgPoolHead, LayerNorm2d, scale_channels


class PoolTokenMixer(nn.Module):
    def __init__(self, channels: int, *, pool_size: int = 3) -> None:
        super().__init__()
        c = int(channels)
        k = int(pool_size)
        self.pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x) - x


class MetaFormerBlock(nn.Module):
    def __init__(self, dim: int, *, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = LayerNorm2d(d)
        self.mix = PoolTokenMixer(d, pool_size=3)
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = LayerNorm2d(d)
        self.mlp = nn.Sequential(
            nn.Conv2d(d, 4 * d, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(4 * d, d, kernel_size=1, bias=True),
        )
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.mix(self.norm1(x)))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class MetaFormerClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int] = (64, 128, 320, 512),
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)
        depths = tuple(int(x) for x in depths)
        total = sum(depths)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=total).tolist()
        dp_iter = iter(dp_rates)

        self.down = nn.ModuleList()
        self.down.append(nn.Sequential(nn.Conv2d(int(in_channels), dims[0], kernel_size=4, stride=4), LayerNorm2d(dims[0])))
        for i in range(3):
            self.down.append(nn.Sequential(LayerNorm2d(dims[i]), nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)))

        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[MetaFormerBlock(dims[i], drop_path=float(next(dp_iter))) for _ in range(depths[i])]))

        self.head = GlobalAvgPoolHead(dims[-1], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        for i in range(4):
            x = self.down[i](x)
            x = self.stages[i](x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "metaformer_tiny": {"dims": (64, 128, 320, 512), "depths": (2, 2, 6, 2)},
    "metaformer_small": {"dims": (64, 128, 320, 512), "depths": (3, 4, 8, 3)},
}


def build_metaformer_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "metaformer_tiny",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MetaFormer variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return MetaFormerClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_metaformer_classifier(in_channels=3, num_classes=10, variant="metaformer_tiny", width_mult=0.5)
    y = m(x)
    print("metaformer_tiny", tuple(y.shape))

