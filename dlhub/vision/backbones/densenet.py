import torch
from torch import nn

from dlhub.vision.backbones._blocks import GlobalAvgPoolHead, scale_channels


class _DenseLayer(nn.Module):
    def __init__(
        self, in_ch: int, growth_rate: int, *, bn_size: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inter = int(bn_size) * int(growth_rate)
        self.norm1 = nn.BatchNorm2d(int(in_ch))
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(int(in_ch), inter, kernel_size=1, bias=False)

        self.norm2 = nn.BatchNorm2d(inter)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter, int(growth_rate), kernel_size=3, padding=1, bias=False)

        self.drop = nn.Dropout2d(p=float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.act1(y)
        y = self.conv1(y)
        y = self.norm2(y)
        y = self.act2(y)
        y = self.conv2(y)
        y = self.drop(y)
        return torch.cat([x, y], dim=1)


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_ch: int,
        growth_rate: int,
        *,
        bn_size: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = int(in_ch)
        for _ in range(int(num_layers)):
            layers.append(
                _DenseLayer(c, int(growth_rate), bn_size=int(bn_size), dropout=float(dropout))
            )
            c += int(growth_rate)
        self.block = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Transition(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(int(in_ch))
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return self.pool(x)


class DenseNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        block_layers: tuple[int, int, int, int],
        growth_rate: int = 32,
        init_channels: int = 64,
        width_mult: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        c0 = scale_channels(int(init_channels), w, min_ch=16, divisor=8)
        g = max(4, int(round(int(growth_rate) * w)))

        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), c0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        c = c0
        b1 = _DenseBlock(block_layers[0], c, g, dropout=float(dropout))
        c = b1.out_channels
        t1 = _Transition(c, c // 2)
        c = c // 2

        b2 = _DenseBlock(block_layers[1], c, g, dropout=float(dropout))
        c = b2.out_channels
        t2 = _Transition(c, c // 2)
        c = c // 2

        b3 = _DenseBlock(block_layers[2], c, g, dropout=float(dropout))
        c = b3.out_channels
        t3 = _Transition(c, c // 2)
        c = c // 2

        b4 = _DenseBlock(block_layers[3], c, g, dropout=float(dropout))
        c = b4.out_channels

        self.features = nn.Sequential(
            b1, t1, b2, t2, b3, t3, b4, nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
        self.head = GlobalAvgPoolHead(c, int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "densenet121": {"layers": (6, 12, 24, 16), "growth_rate": 32, "init": 64},
    "densenet169": {"layers": (6, 12, 32, 32), "growth_rate": 32, "init": 64},
    "densenet201": {"layers": (6, 12, 48, 32), "growth_rate": 32, "init": 64},
    "densenet264": {"layers": (6, 12, 64, 48), "growth_rate": 32, "init": 64},
}


def build_densenet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "densenet121",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DenseNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return DenseNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        block_layers=tuple(map(int, spec["layers"])),
        growth_rate=int(spec["growth_rate"]),
        init_channels=int(spec["init"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["densenet121", "densenet201"]:
        m = build_densenet_classifier(
            in_channels=3, num_classes=10, variant=v, width_mult=0.5, dropout=0.1
        )
        y = m(x)
        print(v, tuple(y.shape))
