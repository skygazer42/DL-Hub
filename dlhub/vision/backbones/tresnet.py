
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, DropPath, GlobalAvgPoolHead, SqueezeExcite, scale_channels


class SpaceToDepth(nn.Module):
    def __init__(self, block_size: int = 2) -> None:
        super().__init__()
        b = int(block_size)
        if b <= 1:
            raise ValueError("block_size must be >= 2")
        self.block_size = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        bs = self.block_size
        if h % bs != 0 or w % bs != 0:
            raise ValueError(f"Input H,W must be divisible by block_size={bs}. Got {h}x{w}")
        x = x.view(b, c, h // bs, bs, w // bs, bs)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * (bs * bs), h // bs, w // bs)


def _conv3x3(in_ch: int, out_ch: int, *, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        int(in_ch),
        int(out_ch),
        kernel_size=3,
        stride=int(stride),
        padding=1,
        groups=int(groups),
        bias=False,
    )


def _conv1x1(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, stride=int(stride), bias=False)


class TResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, out_ch: int, *, stride: int, drop_path: float = 0.0) -> None:
        super().__init__()
        mid = int(out_ch)
        out_exp = int(out_ch) * self.expansion
        self.conv1 = _conv1x1(in_ch, mid, stride=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = _conv3x3(mid, mid, stride=int(stride))
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = _conv1x1(mid, out_exp, stride=1)
        self.bn3 = nn.BatchNorm2d(out_exp)
        self.se = SqueezeExcite(out_exp, se_ratio=0.25)
        self.drop_path = DropPath(float(drop_path))
        self.act = nn.ReLU(inplace=True)

        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != out_exp:
            self.down = nn.Sequential(_conv1x1(in_ch, out_exp, stride=int(stride)), nn.BatchNorm2d(out_exp))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.se(x)
        x = self.drop_path(x)
        if self.down is not None:
            identity = self.down(identity)
        return self.act(x + identity)


class TResNetClassifier(nn.Module):
    """TResNet-inspired: Space-to-Depth stem + SE + stochastic depth (simplified)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        layers: tuple[int, int, int, int],
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        c1 = scale_channels(64, w)
        c2 = scale_channels(128, w)
        c3 = scale_channels(256, w)
        c4 = scale_channels(512, w)

        # Space-to-depth increases channels by 4 and halves resolution.
        self.s2d = SpaceToDepth(block_size=2)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels) * 4, c1, kernel_size=3, stride=1, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        total_blocks = sum(int(x) for x in layers)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=total_blocks).tolist()
        dp_iter = iter(dp_rates)

        self.in_ch = c1
        self.layer1 = self._make_layer(c1, layers[0], stride=1, dp_iter=dp_iter)
        self.layer2 = self._make_layer(c2, layers[1], stride=2, dp_iter=dp_iter)
        self.layer3 = self._make_layer(c3, layers[2], stride=2, dp_iter=dp_iter)
        self.layer4 = self._make_layer(c4, layers[3], stride=2, dp_iter=dp_iter)

        out_dim = c4 * TResNetBottleneck.expansion
        self.head = GlobalAvgPoolHead(out_dim, int(num_classes), dropout=float(dropout))

    def _make_layer(self, out_ch: int, blocks: int, *, stride: int, dp_iter: iter) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(TResNetBottleneck(self.in_ch, int(out_ch), stride=int(stride), drop_path=float(next(dp_iter))))
        self.in_ch = int(out_ch) * TResNetBottleneck.expansion
        for _ in range(int(blocks) - 1):
            layers.append(TResNetBottleneck(self.in_ch, int(out_ch), stride=1, drop_path=float(next(dp_iter))))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.s2d(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "tresnet_m": {"layers": (3, 4, 11, 3)},
    "tresnet_l": {"layers": (4, 5, 18, 3)},
}


def build_tresnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "tresnet_m",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown TResNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return TResNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        layers=tuple(map(int, spec["layers"])),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_tresnet_classifier(in_channels=3, num_classes=10, variant="tresnet_m", width_mult=0.5)
    y = m(x)
    print("tresnet_m", tuple(y.shape))

