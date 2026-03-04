from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


class HyperDepthwiseConv2d(nn.Module):
    """A hypernetwork-generated depthwise conv (per-sample weights)."""

    def __init__(self, channels: int, *, kernel_size: int = 3, hidden: int = 128, stride: int = 1) -> None:
        super().__init__()
        c = int(channels)
        k = int(kernel_size)
        if k <= 0 or k % 2 == 0:
            raise ValueError("kernel_size must be positive odd")
        self.channels = c
        self.k = k
        self.stride = int(stride)
        self.padding = k // 2
        h = int(hidden)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, c * k * k),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {c}")
        w_dyn = self.mlp(self.pool(x)).view(b * c, 1, self.k, self.k)
        xg = x.view(1, b * c, h, w)
        y = F.conv2d(xg, w_dyn, bias=None, stride=self.stride, padding=self.padding, groups=b * c)
        return y.view(b, c, y.shape[-2], y.shape[-1])


class HyperConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, k: int = 3) -> None:
        super().__init__()
        self.pre = ConvBNAct(int(in_ch), int(out_ch), kernel_size=1, stride=1, padding=0, act="relu")
        self.dw = HyperDepthwiseConv2d(int(out_ch), kernel_size=int(k), hidden=max(64, int(out_ch) // 2), stride=int(stride))
        self.bn = nn.BatchNorm2d(int(out_ch))
        self.act = nn.ReLU(inplace=True)
        self.pw = ConvBNAct(int(out_ch), int(out_ch), kernel_size=1, stride=1, padding=0, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.dw(x)
        x = self.act(self.bn(x))
        return self.pw(x)


class HyperConvNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int, int] = (48, 96, 192, 384),
        kernel_size: int = 3,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        chs = tuple(scale_channels(int(c), float(width_mult), min_ch=16, divisor=8) for c in channels)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), chs[0], kernel_size=3, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = HyperConvBlock(chs[0], chs[0], stride=1, k=int(kernel_size))
        self.stage2 = HyperConvBlock(chs[0], chs[1], stride=2, k=int(kernel_size))
        self.stage3 = HyperConvBlock(chs[1], chs[2], stride=2, k=int(kernel_size))
        self.stage4 = HyperConvBlock(chs[2], chs[3], stride=2, k=int(kernel_size))
        self.head = GlobalAvgPoolHead(chs[3], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "hyperconv_tiny": {"channels": (40, 80, 160, 320), "k": 3},
    "hyperconv_base": {"channels": (48, 96, 192, 384), "k": 3},
    "hyperconv_largek": {"channels": (48, 96, 192, 384), "k": 5},
}


def build_hyperconv_net_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "hyperconv_base",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown HyperConvNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return HyperConvNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        channels=tuple(map(int, spec["channels"])),
        kernel_size=int(spec["k"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_hyperconv_net_classifier(in_channels=3, num_classes=10, variant="hyperconv_base", width_mult=0.5)
    y = m(x)
    print("hyperconv_base", tuple(y.shape))

