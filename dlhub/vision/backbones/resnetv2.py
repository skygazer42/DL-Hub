from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


def _conv3x3(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), padding=1, bias=False)


def _conv1x1(in_ch: int, out_ch: int, *, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, stride=int(stride), padding=0, bias=False)


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, *, stride: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(int(in_ch))
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = _conv3x3(in_ch, out_ch, stride=int(stride))

        self.bn2 = nn.BatchNorm2d(int(out_ch))
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_ch, out_ch, stride=1)

        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.down = _conv1x1(in_ch, out_ch, stride=int(stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.bn1(x))
        identity = self.down(y) if self.down is not None else x
        y = self.conv1(y)
        y = self.conv2(self.act2(self.bn2(y)))
        return y + identity


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, out_ch: int, *, stride: int) -> None:
        super().__init__()
        mid = int(out_ch)
        out_exp = int(out_ch) * self.expansion

        self.bn1 = nn.BatchNorm2d(int(in_ch))
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = _conv1x1(in_ch, mid, stride=1)

        self.bn2 = nn.BatchNorm2d(mid)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(mid, mid, stride=int(stride))

        self.bn3 = nn.BatchNorm2d(mid)
        self.act3 = nn.ReLU(inplace=True)
        self.conv3 = _conv1x1(mid, out_exp, stride=1)

        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != out_exp:
            self.down = _conv1x1(in_ch, out_exp, stride=int(stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.bn1(x))
        identity = self.down(y) if self.down is not None else x
        y = self.conv1(y)
        y = self.conv2(self.act2(self.bn2(y)))
        y = self.conv3(self.act3(self.bn3(y)))
        return y + identity


class ResNetV2Classifier(nn.Module):
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
            ConvBNAct(int(in_channels), c1, kernel_size=7, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_ch = c1
        self.layer1 = self._make_layer(block, c1, layers[0], stride=1)
        self.layer2 = self._make_layer(block, c2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, c3, layers[2], stride=2)
        self.layer4 = self._make_layer(block, c4, layers[3], stride=2)

        out_dim = c4 * int(getattr(block, "expansion", 1))
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU(inplace=True)
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
        x = self.act(self.norm(x))
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "resnetv2_18": {"block": PreActBasicBlock, "layers": (2, 2, 2, 2)},
    "resnetv2_34": {"block": PreActBasicBlock, "layers": (3, 4, 6, 3)},
    "resnetv2_50": {"block": PreActBottleneck, "layers": (3, 4, 6, 3)},
    "resnetv2_101": {"block": PreActBottleneck, "layers": (3, 4, 23, 3)},
}


def build_resnetv2_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "resnetv2_50",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ResNetV2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ResNetV2Classifier(
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
    for v in ["resnetv2_18", "resnetv2_50"]:
        m = build_resnetv2_classifier(in_channels=3, num_classes=10, variant=v, width_mult=0.5)
        y = m(x)
        print(v, tuple(y.shape))

