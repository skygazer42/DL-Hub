import torch
from torch import nn

from dlhub.vision.backbones._blocks import GlobalAvgPoolHead, LayerNorm2d, scale_channels


class ConvMLPBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = LayerNorm2d(d)
        self.mix = nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d, bias=False)
        self.norm2 = LayerNorm2d(d)
        self.mlp = nn.Sequential(
            nn.Conv2d(d, 4 * d, kernel_size=1), nn.GELU(), nn.Conv2d(4 * d, d, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvMLPClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int] = (64, 128, 256),
        depths: tuple[int, int, int] = (2, 4, 2),
        patch_size: int = 4,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)
        depths = tuple(int(x) for x in depths)
        p = int(patch_size)
        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), dims[0], kernel_size=p, stride=p), LayerNorm2d(dims[0])
        )
        self.stage1 = nn.Sequential(*[ConvMLPBlock(dims[0]) for _ in range(depths[0])])
        self.down1 = nn.Sequential(
            LayerNorm2d(dims[0]), nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
        )
        self.stage2 = nn.Sequential(*[ConvMLPBlock(dims[1]) for _ in range(depths[1])])
        self.down2 = nn.Sequential(
            LayerNorm2d(dims[1]), nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
        )
        self.stage3 = nn.Sequential(*[ConvMLPBlock(dims[2]) for _ in range(depths[2])])
        self.head = GlobalAvgPoolHead(dims[2], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "convmlp_tiny": {"dims": (64, 128, 256), "depths": (2, 4, 2)},
    "convmlp_small": {"dims": (80, 160, 320), "depths": (2, 6, 2)},
}


def build_convmlp_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "convmlp_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ConvMLP variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ConvMLPClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        patch_size=4,
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_convmlp_classifier(
        in_channels=3, num_classes=10, variant="convmlp_tiny", width_mult=0.5
    )
    y = m(x)
    print("convmlp_tiny", tuple(y.shape))
