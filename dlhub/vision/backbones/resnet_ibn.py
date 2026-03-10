import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


class IBN(nn.Module):
    """Instance-Batch Normalization (IBN-Net).

    Splits channels into two parts:
    - first part: InstanceNorm
    - second part: BatchNorm
    """

    def __init__(self, channels: int, *, ratio: float = 0.5) -> None:
        super().__init__()
        c = int(channels)
        r = float(ratio)
        c1 = int(round(c * r))
        c2 = c - c1
        self.c1 = c1
        self.inorm = nn.InstanceNorm2d(c1, affine=True) if c1 > 0 else nn.Identity()
        self.bnorm = nn.BatchNorm2d(c2) if c2 > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c1 == 0:
            return self.bnorm(x)
        if self.c1 == x.shape[1]:
            return self.inorm(x)
        x1, x2 = torch.split(x, [self.c1, x.shape[1] - self.c1], dim=1)
        return torch.cat([self.inorm(x1), self.bnorm(x2)], dim=1)


def _conv3x3(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), padding=1, bias=False
    )


def _conv1x1(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        int(in_ch), int(out_ch), kernel_size=1, stride=int(stride), padding=0, bias=False
    )


class IBNBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, out_ch: int, *, stride: int, use_ibn: bool) -> None:
        super().__init__()
        mid = int(out_ch)
        out_exp = int(out_ch) * self.expansion
        self.conv1 = _conv1x1(in_ch, mid, stride=1)
        self.norm1 = IBN(mid) if bool(use_ibn) else nn.BatchNorm2d(mid)
        self.conv2 = _conv3x3(mid, mid, stride=int(stride))
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = _conv1x1(mid, out_exp, stride=1)
        self.bn3 = nn.BatchNorm2d(out_exp)
        self.act = nn.ReLU(inplace=True)
        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != out_exp:
            self.down = nn.Sequential(
                _conv1x1(in_ch, out_exp, stride=int(stride)), nn.BatchNorm2d(out_exp)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.down is not None:
            identity = self.down(identity)
        return self.act(x + identity)


class ResNetIBNClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        layers: tuple[int, int, int, int],
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        c1 = scale_channels(64, w)
        c2 = scale_channels(128, w)
        c3 = scale_channels(256, w)
        c4 = scale_channels(512, w)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), c1, kernel_size=7, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_ch = c1
        self.layer1 = self._make_layer(c1, layers[0], stride=1, use_ibn=True)
        self.layer2 = self._make_layer(c2, layers[1], stride=2, use_ibn=True)
        self.layer3 = self._make_layer(c3, layers[2], stride=2, use_ibn=False)
        self.layer4 = self._make_layer(c4, layers[3], stride=2, use_ibn=False)
        out_dim = c4 * IBNBottleneck.expansion
        self.head = GlobalAvgPoolHead(out_dim, int(num_classes), dropout=float(dropout))

    def _make_layer(self, out_ch: int, blocks: int, *, stride: int, use_ibn: bool) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(
            IBNBottleneck(self.in_ch, int(out_ch), stride=int(stride), use_ibn=bool(use_ibn))
        )
        self.in_ch = int(out_ch) * IBNBottleneck.expansion
        for _ in range(int(blocks) - 1):
            layers.append(IBNBottleneck(self.in_ch, int(out_ch), stride=1, use_ibn=bool(use_ibn)))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "resnet_ibn50": {"layers": (3, 4, 6, 3)},
    "resnet_ibn101": {"layers": (3, 4, 23, 3)},
}


def build_resnet_ibn_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "resnet_ibn50",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ResNet-IBN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ResNetIBNClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        layers=tuple(map(int, spec["layers"])),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_resnet_ibn_classifier(
        in_channels=3, num_classes=10, variant="resnet_ibn50", width_mult=0.5
    )
    y = m(x)
    print("resnet_ibn50", tuple(y.shape))
