
import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath, GlobalAvgPoolHead, LayerNorm2d, scale_channels


class FocalModulation(nn.Module):
    """Focal modulation (simplified).

    Multi-scale depthwise context -> gating -> modulate query.
    """

    def __init__(self, dim: int, *, focal_levels: int = 3, kernel_base: int = 3) -> None:
        super().__init__()
        d = int(dim)
        lv = int(focal_levels)
        self.q = nn.Conv2d(d, d, kernel_size=1, bias=True)
        self.ctx = nn.ModuleList()
        for i in range(lv):
            k = int(kernel_base) + 2 * i
            self.ctx.append(nn.Conv2d(d, d, kernel_size=k, padding=k // 2, groups=d, bias=False))
        self.gate = nn.Sequential(nn.Conv2d(d, d, kernel_size=1, bias=True), nn.Sigmoid())
        self.proj = nn.Conv2d(d, d, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        ctx = 0.0
        for c in self.ctx:
            ctx = ctx + c(x)
        ctx = ctx / float(len(self.ctx))
        y = q * self.gate(ctx)
        return self.proj(y)


class FocalNetBlock(nn.Module):
    def __init__(self, dim: int, *, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = LayerNorm2d(d)
        self.fm = FocalModulation(d, focal_levels=3, kernel_base=3)
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = LayerNorm2d(d)
        self.mlp = nn.Sequential(
            nn.Conv2d(d, 4 * d, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(4 * d, d, kernel_size=1, bias=True),
        )
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.fm(self.norm1(x)))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class FocalNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
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
            self.stages.append(nn.Sequential(*[FocalNetBlock(dims[i], drop_path=float(next(dp_iter))) for _ in range(depths[i])]))
        self.head = GlobalAvgPoolHead(dims[-1], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        for i in range(4):
            x = self.down[i](x)
            x = self.stages[i](x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "focalnet_tiny": {"dims": (96, 192, 384, 768), "depths": (2, 2, 6, 2)},
    "focalnet_small": {"dims": (96, 192, 384, 768), "depths": (2, 2, 12, 2)},
}


def build_focalnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "focalnet_tiny",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FocalNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return FocalNetClassifier(
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
    m = build_focalnet_classifier(in_channels=3, num_classes=10, variant="focalnet_tiny", width_mult=0.5)
    y = m(x)
    print("focalnet_tiny", tuple(y.shape))

