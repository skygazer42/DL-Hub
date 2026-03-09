
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


def _conv3x3(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), padding=1, bias=False)


def _conv1x1(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, stride=int(stride), padding=0, bias=False)


class XBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, *, stride: int) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride=int(stride))
        self.bn1 = nn.BatchNorm2d(int(out_ch))
        self.act = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_ch, out_ch, stride=1)
        self.bn2 = nn.BatchNorm2d(int(out_ch))

        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.down = nn.Sequential(_conv1x1(in_ch, out_ch, stride=int(stride)), nn.BatchNorm2d(int(out_ch)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.down is not None:
            identity = self.down(identity)
        return self.act(x + identity)


class XBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, out_ch: int, *, stride: int) -> None:
        super().__init__()
        mid = int(out_ch)
        out_exp = int(out_ch) * self.expansion
        self.conv1 = _conv1x1(in_ch, mid, stride=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = _conv3x3(mid, mid, stride=int(stride))
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = _conv1x1(mid, out_exp, stride=1)
        self.bn3 = nn.BatchNorm2d(out_exp)
        self.act = nn.ReLU(inplace=True)

        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != out_exp:
            self.down = nn.Sequential(_conv1x1(in_ch, out_exp, stride=int(stride)), nn.BatchNorm2d(out_exp))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.down is not None:
            identity = self.down(identity)
        return self.act(x + identity)


class XResNetClassifier(nn.Module):
    """xResNet-style stem (3x3,3x3,3x3) + ResNet blocks (simplified)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        layers: tuple[int, int, int, int],
        block: type[nn.Module],
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
            ConvBNAct(int(in_channels), c1 // 2, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(c1 // 2, c1 // 2, kernel_size=3, stride=1, act="relu"),
            ConvBNAct(c1 // 2, c1, kernel_size=3, stride=1, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.in_ch = c1
        self.layer1 = self._make_layer(block, c1, layers[0], stride=1)
        self.layer2 = self._make_layer(block, c2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, c3, layers[2], stride=2)
        self.layer4 = self._make_layer(block, c4, layers[3], stride=2)

        out_dim = c4 * int(getattr(block, "expansion", 1))
        self.head = GlobalAvgPoolHead(out_dim, int(num_classes), dropout=float(dropout))

    def _make_layer(self, block: type[nn.Module], out_ch: int, blocks: int, *, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(block(self.in_ch, int(out_ch), stride=int(stride)))
        self.in_ch = int(out_ch) * int(getattr(block, "expansion", 1))
        for _ in range(int(blocks) - 1):
            layers.append(block(self.in_ch, int(out_ch), stride=1))
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
    "xresnet18": {"block": XBasicBlock, "layers": (2, 2, 2, 2)},
    "xresnet34": {"block": XBasicBlock, "layers": (3, 4, 6, 3)},
    "xresnet50": {"block": XBottleneck, "layers": (3, 4, 6, 3)},
}


def build_xresnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "xresnet18",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown XResNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return XResNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        layers=tuple(map(int, spec["layers"])),
        block=spec["block"],
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["xresnet18", "xresnet50"]:
        m = build_xresnet_classifier(in_channels=3, num_classes=10, variant=v, width_mult=0.5)
        y = m(x)
        print(v, tuple(y.shape))

