
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


class InceptionV2Block(nn.Module):
    """Inception-v2 / BN-Inception-ish module (simplified)."""

    def __init__(self, in_ch: int, *, c1: int, c3: int, c5: int, pool: int) -> None:
        super().__init__()
        c_in = int(in_ch)
        self.b1 = ConvBNAct(c_in, int(c1), kernel_size=1, stride=1, padding=0, act="relu")

        self.b2 = nn.Sequential(
            ConvBNAct(c_in, int(c3), kernel_size=1, stride=1, padding=0, act="relu"),
            ConvBNAct(int(c3), int(c3), kernel_size=3, stride=1, act="relu"),
        )

        self.b3 = nn.Sequential(
            ConvBNAct(c_in, int(c5), kernel_size=1, stride=1, padding=0, act="relu"),
            ConvBNAct(int(c5), int(c5), kernel_size=3, stride=1, act="relu"),
            ConvBNAct(int(c5), int(c5), kernel_size=3, stride=1, act="relu"),
        )

        self.b4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNAct(c_in, int(pool), kernel_size=1, stride=1, padding=0, act="relu"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], dim=1)


class InceptionV2Classifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int] = (64, 128, 256),
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c1, c2, c3 = (scale_channels(int(c), float(width_mult), min_ch=16, divisor=8) for c in channels)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), c1, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(c1, c1, kernel_size=3, stride=1, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.incept1 = InceptionV2Block(c1, c1=c1 // 2, c3=c1 // 2, c5=c1 // 4, pool=c1 // 4)
        out1 = (c1 // 2) + (c1 // 2) + (c1 // 4) + (c1 // 4)
        self.down2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.incept2 = InceptionV2Block(out1, c1=c2 // 2, c3=c2 // 2, c5=c2 // 4, pool=c2 // 4)
        out2 = (c2 // 2) + (c2 // 2) + (c2 // 4) + (c2 // 4)
        self.down3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.incept3 = InceptionV2Block(out2, c1=c3 // 2, c3=c3 // 2, c5=c3 // 4, pool=c3 // 4)
        out3 = (c3 // 2) + (c3 // 2) + (c3 // 4) + (c3 // 4)

        self.head = GlobalAvgPoolHead(out3, int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.incept1(x)
        x = self.down2(x)
        x = self.incept2(x)
        x = self.down3(x)
        x = self.incept3(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "inception_v2_tiny": {"channels": (48, 96, 192)},
    "inception_v2_base": {"channels": (64, 128, 256)},
}


def build_inception_v2_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "inception_v2_base",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Inception-v2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return InceptionV2Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        channels=tuple(map(int, spec["channels"])),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_inception_v2_classifier(in_channels=3, num_classes=10, variant="inception_v2_tiny", width_mult=0.5)
    y = m(x)
    print("inception_v2_tiny", tuple(y.shape))
