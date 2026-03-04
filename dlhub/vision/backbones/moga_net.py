from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, DropPath, GlobalAvgPoolHead, scale_channels


class MultiOrderGatedAggregation(nn.Module):
    """MogaNet-style multi-order gated aggregation (simplified)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.dw1 = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        self.dw2 = nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2, groups=c, bias=False)
        self.dw3 = nn.Conv2d(c, c, kernel_size=3, padding=3, dilation=3, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.gate = nn.Sequential(nn.Conv2d(c, c, kernel_size=1, bias=True), nn.Sigmoid())
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (self.dw1(x) + self.dw2(x) + self.dw3(x)) / 3.0
        y = self.pw(y)
        y = self.bn(y)
        return y * self.gate(x)


class MogaBlock(nn.Module):
    def __init__(self, dim: int, *, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.agg = MultiOrderGatedAggregation(d)
        self.mlp = nn.Sequential(
            nn.Conv2d(d, 4 * d, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(4 * d, d, kernel_size=1, bias=True),
        )
        self.drop_path = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.agg(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class MogaNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int] = (64, 128, 256, 512),
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
        self.down.append(nn.Sequential(nn.Conv2d(int(in_channels), dims[0], kernel_size=4, stride=4), nn.BatchNorm2d(dims[0])))
        for i in range(3):
            self.down.append(nn.Sequential(nn.BatchNorm2d(dims[i]), nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)))

        self.stages = nn.ModuleList(
            [nn.Sequential(*[MogaBlock(dims[i], drop_path=float(next(dp_iter))) for _ in range(depths[i])]) for i in range(4)]
        )
        self.head = GlobalAvgPoolHead(dims[-1], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        for i in range(4):
            x = self.down[i](x)
            x = self.stages[i](x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "moganet_tiny": {"dims": (64, 128, 256, 512), "depths": (1, 1, 4, 1)},
    "moganet_base": {"dims": (64, 128, 256, 512), "depths": (2, 2, 6, 2)},
    "moganet_large": {"dims": (96, 192, 384, 768), "depths": (2, 2, 8, 2)},
}


def build_moga_net_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "moganet_base",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MogaNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return MogaNetClassifier(
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
    m = build_moga_net_classifier(in_channels=3, num_classes=10, variant="moganet_tiny", width_mult=0.5)
    y = m(x)
    print("moganet_tiny", tuple(y.shape))

