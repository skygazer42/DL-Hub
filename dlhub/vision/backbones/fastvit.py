
import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath, GlobalAvgPoolHead, LayerNorm2d, scale_channels


class FastViTBlock(nn.Module):
    """FastViT-ish block: large kernel depthwise conv + pointwise MLP (simplified)."""

    def __init__(self, dim: int, *, kernel_size: int = 11, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        k = int(kernel_size)
        self.dw = nn.Conv2d(d, d, kernel_size=k, padding=k // 2, groups=d, bias=False)
        self.bn = nn.BatchNorm2d(d)
        self.pw1 = nn.Conv2d(d, 4 * d, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * d, d, kernel_size=1, bias=True)
        self.dp = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.bn(self.dw(x))
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.dp(x)
        return identity + x


class FastViTClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int] = (64, 128, 256, 512),
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        kernel_size: int = 11,
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

        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = [FastViTBlock(dims[i], kernel_size=int(kernel_size), drop_path=float(next(dp_iter))) for _ in range(depths[i])]
            self.stages.append(nn.Sequential(*blocks))

        self.head = GlobalAvgPoolHead(dims[-1], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        for i in range(4):
            x = self.down[i](x)
            x = self.stages[i](x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "fastvit_t8": {"dims": (64, 128, 256, 512), "depths": (2, 2, 6, 2), "k": 11},
    "fastvit_s12": {"dims": (80, 160, 320, 640), "depths": (2, 2, 6, 2), "k": 13},
}


def build_fastvit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "fastvit_t8",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FastViT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return FastViTClassifier(
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
    m = build_fastvit_classifier(in_channels=3, num_classes=10, variant="fastvit_t8", width_mult=0.5)
    y = m(x)
    print("fastvit_t8", tuple(y.shape))

