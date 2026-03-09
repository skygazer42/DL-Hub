
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, DropPath, GlobalAvgPoolHead, scale_channels


class SparseLargeKernel(nn.Module):
    """SLaK-like factorized large kernel depthwise conv (1xK then Kx1)."""

    def __init__(self, channels: int, *, kernel_size: int = 31) -> None:
        super().__init__()
        c = int(channels)
        k = int(kernel_size)
        if k <= 0 or k % 2 == 0:
            raise ValueError("kernel_size must be positive odd")
        self.dw1 = nn.Conv2d(c, c, kernel_size=(1, k), padding=(0, k // 2), groups=c, bias=False)
        self.dw2 = nn.Conv2d(c, c, kernel_size=(k, 1), padding=(k // 2, 0), groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw1(x)
        x = self.dw2(x)
        return self.bn(x)


class SLaKBlock(nn.Module):
    def __init__(self, dim: int, *, kernel_size: int = 31, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.lk = SparseLargeKernel(d, kernel_size=int(kernel_size))
        self.pw1 = nn.Conv2d(d, 4 * d, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * d, d, kernel_size=1, bias=True)
        self.drop_path = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.lk(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.drop_path(x)
        return identity + x


class SLaKClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int] = (64, 128, 256, 512),
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        kernel_size: int = 31,
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult)) for d in dims)
        depths = tuple(int(x) for x in depths)
        total = sum(depths)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=total).tolist()
        dp_iter = iter(dp_rates)

        self.down = nn.ModuleList()
        self.down.append(nn.Sequential(nn.Conv2d(int(in_channels), dims[0], kernel_size=4, stride=4), nn.BatchNorm2d(dims[0])))
        for i in range(3):
            self.down.append(nn.Sequential(nn.BatchNorm2d(dims[i]), nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)))

        self.stages = nn.ModuleList(
            [
                nn.Sequential(*[SLaKBlock(dims[i], kernel_size=int(kernel_size), drop_path=float(next(dp_iter))) for _ in range(depths[i])])
                for i in range(4)
            ]
        )
        self.head = GlobalAvgPoolHead(dims[-1], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        for i in range(4):
            x = self.down[i](x)
            x = self.stages[i](x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "slak_tiny": {"dims": (64, 128, 256, 512), "depths": (1, 1, 4, 1), "k": 31},
    "slak_base": {"dims": (64, 128, 256, 512), "depths": (2, 2, 6, 2), "k": 31},
    "slak_hugek": {"dims": (64, 128, 256, 512), "depths": (2, 2, 6, 2), "k": 51},
}


def build_slak_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "slak_base",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SLaK variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return SLaKClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        kernel_size=int(spec["k"]),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_slak_classifier(in_channels=3, num_classes=10, variant="slak_tiny", width_mult=0.5)
    y = m(x)
    print("slak_tiny", tuple(y.shape))

