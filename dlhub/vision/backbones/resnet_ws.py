import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


class WSConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization (WS)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        w = w - w.mean(dim=(1, 2, 3), keepdim=True)
        std = w.flatten(1).std(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-5
        w = w / std
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def _gn(channels: int, *, groups: int = 32) -> nn.GroupNorm:
    c = int(channels)
    g = min(int(groups), c)
    while c % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, c)


def _ws3x3(in_ch: int, out_ch: int, *, stride: int = 1) -> WSConv2d:
    return WSConv2d(
        int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), padding=1, bias=False
    )


def _ws1x1(in_ch: int, out_ch: int, *, stride: int = 1) -> WSConv2d:
    return WSConv2d(
        int(in_ch), int(out_ch), kernel_size=1, stride=int(stride), padding=0, bias=False
    )


class WSBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, out_ch: int, *, stride: int, groups: int = 32) -> None:
        super().__init__()
        mid = int(out_ch)
        out_exp = int(out_ch) * self.expansion
        self.conv1 = _ws1x1(in_ch, mid, stride=1)
        self.norm1 = _gn(mid, groups=int(groups))
        self.conv2 = _ws3x3(mid, mid, stride=int(stride))
        self.norm2 = _gn(mid, groups=int(groups))
        self.conv3 = _ws1x1(mid, out_exp, stride=1)
        self.norm3 = _gn(out_exp, groups=int(groups))
        self.act = nn.ReLU(inplace=True)
        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != out_exp:
            self.down = nn.Sequential(
                _ws1x1(in_ch, out_exp, stride=int(stride)), _gn(out_exp, groups=int(groups))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        if self.down is not None:
            identity = self.down(identity)
        return self.act(x + identity)


class ResNetWSClassifier(nn.Module):
    """ResNet with Weight Standardization (WS) + GroupNorm."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        layers: tuple[int, int, int, int],
        width_mult: float = 1.0,
        groups: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        c1 = scale_channels(64, w)
        c2 = scale_channels(128, w)
        c3 = scale_channels(256, w)
        c4 = scale_channels(512, w)
        self.groups = int(groups)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), c1, kernel_size=7, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.in_ch = c1
        self.layer1 = self._make_layer(c1, layers[0], stride=1)
        self.layer2 = self._make_layer(c2, layers[1], stride=2)
        self.layer3 = self._make_layer(c3, layers[2], stride=2)
        self.layer4 = self._make_layer(c4, layers[3], stride=2)

        out_dim = c4 * WSBottleneck.expansion
        self.head = GlobalAvgPoolHead(out_dim, int(num_classes), dropout=float(dropout))

    def _make_layer(self, out_ch: int, blocks: int, *, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(WSBottleneck(self.in_ch, int(out_ch), stride=int(stride), groups=self.groups))
        self.in_ch = int(out_ch) * WSBottleneck.expansion
        for _ in range(int(blocks) - 1):
            layers.append(WSBottleneck(self.in_ch, int(out_ch), stride=1, groups=self.groups))
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
    "resnet_ws50": {"layers": (3, 4, 6, 3)},
    "resnet_ws101": {"layers": (3, 4, 23, 3)},
}


def build_resnet_ws_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "resnet_ws50",
    width_mult: float = 1.0,
    groups: int = 32,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ResNet-WS variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ResNetWSClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        layers=tuple(map(int, spec["layers"])),
        width_mult=float(width_mult),
        groups=int(groups),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_resnet_ws_classifier(
        in_channels=3, num_classes=10, variant="resnet_ws50", width_mult=0.5
    )
    y = m(x)
    print("resnet_ws50", tuple(y.shape))
