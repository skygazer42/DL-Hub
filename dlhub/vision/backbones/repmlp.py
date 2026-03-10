import torch
from torch import nn

from dlhub.vision.backbones._blocks import GlobalAvgPoolHead, LayerNorm2d, scale_channels


class RepMLPBlock(nn.Module):
    """RepMLP-style block (simplified).

    Mixes local depthwise conv and global channel mixing.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = LayerNorm2d(d)
        self.local = nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d, bias=False)
        self.local_bn = nn.BatchNorm2d(d)
        self.norm2 = LayerNorm2d(d)
        self.global_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(d, d), nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(d, 4 * d, kernel_size=1), nn.GELU(), nn.Conv2d(4 * d, d, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.local_bn(self.local(self.norm1(x)))
        x = x + y
        g = self.global_fc(self.norm2(x)).view(x.shape[0], x.shape[1], 1, 1)
        x = x * (1.0 + g)
        x = x + self.mlp(x)
        return x


class RepMLPClassifier(nn.Module):
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
        self.patch = nn.Sequential(
            nn.Conv2d(int(in_channels), d, kernel_size=p, stride=p), LayerNorm2d(d)
        )
        self.blocks = nn.Sequential(*[RepMLPBlock(d) for _ in range(int(depth))])
        self.head = GlobalAvgPoolHead(d, int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x)
        x = self.blocks(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "repmlp_tiny": {"dim": 192, "depth": 10, "patch": 4},
    "repmlp_small": {"dim": 256, "depth": 12, "patch": 4},
}


def build_repmlp_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "repmlp_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RepMLP variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return RepMLPClassifier(
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
    m = build_repmlp_classifier(
        in_channels=3, num_classes=10, variant="repmlp_tiny", width_mult=0.5
    )
    y = m(x)
    print("repmlp_tiny", tuple(y.shape))
