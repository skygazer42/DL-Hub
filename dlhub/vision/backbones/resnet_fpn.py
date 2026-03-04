from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


def _conv3x3(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), padding=1, bias=False)


def _conv1x1(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, stride=int(stride), padding=0, bias=False)


class FPNBottleneck(nn.Module):
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


class FeaturePyramid(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int, int], out_channels: int = 256) -> None:
        super().__init__()
        c2, c3, c4, c5 = (int(x) for x in in_channels)
        o = int(out_channels)
        self.l2 = _conv1x1(c2, o)
        self.l3 = _conv1x1(c3, o)
        self.l4 = _conv1x1(c4, o)
        self.l5 = _conv1x1(c5, o)

        self.s2 = _conv3x3(o, o)
        self.s3 = _conv3x3(o, o)
        self.s4 = _conv3x3(o, o)
        self.s5 = _conv3x3(o, o)

    def forward(self, feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        c2, c3, c4, c5 = feats
        p5 = self.l5(c5)
        p4 = self.l4(c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.l2(c2) + nn.functional.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.s2(p2)
        p3 = self.s3(p3)
        p4 = self.s4(p4)
        p5 = self.s5(p5)
        return (p2, p3, p4, p5)


class ResNetFPNClassifier(nn.Module):
    """ResNet backbone + FPN top-down merge (classification head on P5)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        layers: tuple[int, int, int, int],
        width_mult: float = 1.0,
        fpn_dim: int = 256,
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
        self.layer1 = self._make_layer(c1, layers[0], stride=1)
        self.layer2 = self._make_layer(c2, layers[1], stride=2)
        self.layer3 = self._make_layer(c3, layers[2], stride=2)
        self.layer4 = self._make_layer(c4, layers[3], stride=2)

        c2_out = c1 * FPNBottleneck.expansion
        c3_out = c2 * FPNBottleneck.expansion
        c4_out = c3 * FPNBottleneck.expansion
        c5_out = c4 * FPNBottleneck.expansion
        self.fpn = FeaturePyramid((c2_out, c3_out, c4_out, c5_out), out_channels=int(fpn_dim))
        self.head = GlobalAvgPoolHead(int(fpn_dim), int(num_classes), dropout=float(dropout))

    def _make_layer(self, out_ch: int, blocks: int, *, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(FPNBottleneck(self.in_ch, int(out_ch), stride=int(stride)))
        self.in_ch = int(out_ch) * FPNBottleneck.expansion
        for _ in range(int(blocks) - 1):
            layers.append(FPNBottleneck(self.in_ch, int(out_ch), stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p2, p3, p4, p5 = self.fpn((c2, c3, c4, c5))
        return self.head(p5)


_VARIANTS: dict[str, dict] = {
    "resnet_fpn50": {"layers": (3, 4, 6, 3)},
    "resnet_fpn101": {"layers": (3, 4, 23, 3)},
}


def build_resnet_fpn_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "resnet_fpn50",
    width_mult: float = 1.0,
    fpn_dim: int = 256,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ResNet-FPN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ResNetFPNClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        layers=tuple(map(int, spec["layers"])),
        width_mult=float(width_mult),
        fpn_dim=int(fpn_dim),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_resnet_fpn_classifier(in_channels=3, num_classes=10, variant="resnet_fpn50", width_mult=0.5)
    y = m(x)
    print("resnet_fpn50", tuple(y.shape))

