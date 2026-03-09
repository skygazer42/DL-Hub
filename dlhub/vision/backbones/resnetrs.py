
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, DropPath, GlobalAvgPoolHead, SqueezeExcite, scale_channels


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


class ResNetRSBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, out_ch: int, *, stride: int, se_ratio: float = 0.25, drop_path: float = 0.0) -> None:
        super().__init__()
        mid = int(out_ch)
        out_exp = int(out_ch) * self.expansion
        self.conv1 = _conv1x1(in_ch, mid, stride=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = _conv3x3(mid, mid, stride=int(stride))
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = _conv1x1(mid, out_exp, stride=1)
        self.bn3 = nn.BatchNorm2d(out_exp)
        self.se = SqueezeExcite(out_exp, se_ratio=float(se_ratio)) if float(se_ratio) > 0 else nn.Identity()
        self.drop_path = DropPath(float(drop_path))
        self.act = nn.ReLU(inplace=True)

        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != out_exp:
            ops: list[nn.Module] = []
            if int(stride) != 1:
                ops.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
                stride = 1
            ops.append(_conv1x1(in_ch, out_exp, stride=int(stride)))
            ops.append(nn.BatchNorm2d(out_exp))
            self.down = nn.Sequential(*ops)

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


class ResNetRSClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        layers: tuple[int, int, int, int],
        width_mult: float = 1.0,
        se_ratio: float = 0.25,
        drop_path: float = 0.1,
        dropout: float = 0.2,
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

        total_blocks = sum(int(x) for x in layers)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=total_blocks).tolist()
        dp_iter = iter(dp_rates)

        self.in_ch = c1
        self.layer1 = self._make_layer(c1, layers[0], stride=1, se_ratio=float(se_ratio), dp_iter=dp_iter)
        self.layer2 = self._make_layer(c2, layers[1], stride=2, se_ratio=float(se_ratio), dp_iter=dp_iter)
        self.layer3 = self._make_layer(c3, layers[2], stride=2, se_ratio=float(se_ratio), dp_iter=dp_iter)
        self.layer4 = self._make_layer(c4, layers[3], stride=2, se_ratio=float(se_ratio), dp_iter=dp_iter)

        out_dim = c4 * ResNetRSBottleneck.expansion
        self.head = GlobalAvgPoolHead(out_dim, int(num_classes), dropout=float(dropout))

    def _make_layer(
        self,
        out_ch: int,
        blocks: int,
        *,
        stride: int,
        se_ratio: float,
        dp_iter: iter,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(
            ResNetRSBottleneck(self.in_ch, int(out_ch), stride=int(stride), se_ratio=float(se_ratio), drop_path=float(next(dp_iter)))
        )
        self.in_ch = int(out_ch) * ResNetRSBottleneck.expansion
        for _ in range(int(blocks) - 1):
            layers.append(
                ResNetRSBottleneck(self.in_ch, int(out_ch), stride=1, se_ratio=float(se_ratio), drop_path=float(next(dp_iter)))
            )
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
    "resnetrs50": {"layers": (3, 4, 6, 3)},
    "resnetrs101": {"layers": (3, 4, 23, 3)},
    "resnetrs152": {"layers": (3, 8, 36, 3)},
}


def build_resnetrs_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "resnetrs50",
    width_mult: float = 1.0,
    se_ratio: float = 0.25,
    drop_path: float = 0.1,
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ResNet-RS variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ResNetRSClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        layers=tuple(map(int, spec["layers"])),
        width_mult=float(width_mult),
        se_ratio=float(se_ratio),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_resnetrs_classifier(in_channels=3, num_classes=10, variant="resnetrs50", width_mult=0.5)
    y = m(x)
    print("resnetrs50", tuple(y.shape))

