from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


class CondConv2d(nn.Module):
    """Conditionally parameterized conv (CondConv) with K experts."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        num_experts: int = 4,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        if padding is None:
            padding = k // 2
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.k = k
        self.stride = int(stride)
        self.padding = int(padding)
        self.num_experts = int(num_experts)

        self.weight = nn.Parameter(torch.randn(self.num_experts, self.out_ch, self.in_ch, k, k) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_experts, self.out_ch))

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.in_ch, self.num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        b, c, h, w = x.shape
        if c != self.in_ch:
            raise ValueError(f"Expected in_ch={self.in_ch}, got {c}")

        g = torch.softmax(self.gate(x), dim=-1)  # (B, K)
        w = torch.einsum("bk,kocij->bocij", g, self.weight)  # (B, O, C, k, k)
        b0 = torch.einsum("bk,ko->bo", g, self.bias)  # (B, O)

        # group-conv trick for per-sample weights
        x = x.reshape(1, b * c, h, w)
        w = w.reshape(b * self.out_ch, self.in_ch, self.k, self.k)
        y = F.conv2d(x, w, bias=None, stride=self.stride, padding=self.padding, groups=b)
        y = y.reshape(b, self.out_ch, y.shape[-2], y.shape[-1])
        y = y + b0[:, :, None, None]
        return y


class CondConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int, num_experts: int) -> None:
        super().__init__()
        self.conv = CondConv2d(int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), num_experts=int(num_experts))
        self.bn = nn.BatchNorm2d(int(out_ch))
        self.act = nn.ReLU(inplace=True)
        self.pw = ConvBNAct(int(out_ch), int(out_ch), kernel_size=1, stride=1, padding=0, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(self.bn(x))
        return self.pw(x)


class CondConvNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int, int] = (48, 96, 192, 384),
        num_experts: int = 4,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        chs = tuple(scale_channels(int(c), float(width_mult), min_ch=16, divisor=8) for c in channels)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), chs[0], kernel_size=3, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = CondConvBlock(chs[0], chs[0], stride=1, num_experts=int(num_experts))
        self.stage2 = CondConvBlock(chs[0], chs[1], stride=2, num_experts=int(num_experts))
        self.stage3 = CondConvBlock(chs[1], chs[2], stride=2, num_experts=int(num_experts))
        self.stage4 = CondConvBlock(chs[2], chs[3], stride=2, num_experts=int(num_experts))
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
    "condconv_tiny": {"channels": (40, 80, 160, 320), "experts": 4},
    "condconv_base": {"channels": (48, 96, 192, 384), "experts": 4},
    "condconv_wide": {"channels": (64, 128, 256, 512), "experts": 8},
}


def build_condconv_net_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "condconv_base",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CondConvNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return CondConvNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        channels=tuple(map(int, spec["channels"])),
        num_experts=int(spec["experts"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_condconv_net_classifier(in_channels=3, num_classes=10, variant="condconv_tiny", width_mult=0.5)
    y = m(x)
    print("condconv_tiny", tuple(y.shape))

