
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


class DropBlock2D(nn.Module):
    """DropBlock (2D) regularization.

    This is a compact implementation intended for training-time regularization.
    """

    def __init__(self, p: float = 0.0, block_size: int = 7) -> None:
        super().__init__()
        self.p = float(p)
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        b, c, h, w = x.shape
        bs = self.block_size
        if bs > h or bs > w:
            return x

        # gamma per DropBlock paper
        gamma = self.p * (h * w) / (bs * bs) / ((h - bs + 1) * (w - bs + 1))
        mask = torch.empty((b, 1, h - bs + 1, w - bs + 1), device=x.device, dtype=x.dtype).bernoulli_(gamma)
        # pad to full size then maxpool to create blocks
        mask = nn.functional.pad(mask, (bs // 2, bs - 1 - bs // 2, bs // 2, bs - 1 - bs // 2))
        block_mask = nn.functional.max_pool2d(mask, kernel_size=bs, stride=1, padding=bs // 2)
        block_mask = 1.0 - block_mask
        # normalize
        denom = block_mask.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return x * block_mask / denom


def _conv3x3(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), padding=1, bias=False)


def _conv1x1(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, stride=int(stride), padding=0, bias=False)


class DropBlockBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, out_ch: int, *, stride: int, dropblock: DropBlock2D) -> None:
        super().__init__()
        mid = int(out_ch)
        out_exp = int(out_ch) * self.expansion
        self.conv1 = _conv1x1(in_ch, mid, stride=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = _conv3x3(mid, mid, stride=int(stride))
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = _conv1x1(mid, out_exp, stride=1)
        self.bn3 = nn.BatchNorm2d(out_exp)
        self.dropblock = dropblock
        self.act = nn.ReLU(inplace=True)
        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != out_exp:
            self.down = nn.Sequential(_conv1x1(in_ch, out_exp, stride=int(stride)), nn.BatchNorm2d(out_exp))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.dropblock(x)
        if self.down is not None:
            identity = self.down(identity)
        return self.act(x + identity)


class ResNetDropBlockClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        layers: tuple[int, int, int, int],
        width_mult: float = 1.0,
        dropblock_p: float = 0.1,
        dropblock_size: int = 7,
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

        self.dropblock = DropBlock2D(p=float(dropblock_p), block_size=int(dropblock_size))
        self.in_ch = c1
        self.layer1 = self._make_layer(c1, layers[0], stride=1)
        self.layer2 = self._make_layer(c2, layers[1], stride=2)
        self.layer3 = self._make_layer(c3, layers[2], stride=2)
        self.layer4 = self._make_layer(c4, layers[3], stride=2)

        out_dim = c4 * DropBlockBottleneck.expansion
        self.head = GlobalAvgPoolHead(out_dim, int(num_classes), dropout=float(dropout))

    def _make_layer(self, out_ch: int, blocks: int, *, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(DropBlockBottleneck(self.in_ch, int(out_ch), stride=int(stride), dropblock=self.dropblock))
        self.in_ch = int(out_ch) * DropBlockBottleneck.expansion
        for _ in range(int(blocks) - 1):
            layers.append(DropBlockBottleneck(self.in_ch, int(out_ch), stride=1, dropblock=self.dropblock))
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
    "resnet_dropblock50": {"layers": (3, 4, 6, 3)},
    "resnet_dropblock101": {"layers": (3, 4, 23, 3)},
}


def build_resnet_dropblock_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "resnet_dropblock50",
    width_mult: float = 1.0,
    dropblock_p: float = 0.1,
    dropblock_size: int = 7,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ResNet-DropBlock variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ResNetDropBlockClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        layers=tuple(map(int, spec["layers"])),
        width_mult=float(width_mult),
        dropblock_p=float(dropblock_p),
        dropblock_size=int(dropblock_size),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_resnet_dropblock_classifier(in_channels=3, num_classes=10, variant="resnet_dropblock50", width_mult=0.5)
    y = m(x)
    print("resnet_dropblock50", tuple(y.shape))

