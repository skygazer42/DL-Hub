
import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


class DynamicConv2d(nn.Module):
    """Dynamic convolution via attention over expert kernels (DyConv-style)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        num_experts: int = 4,
        temperature: float = 30.0,
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
        self.temperature = float(temperature)

        self.weight = nn.Parameter(torch.randn(self.num_experts, self.out_ch, self.in_ch, k, k) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.num_experts, self.out_ch))

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.in_ch, max(8, self.in_ch // 4), kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, self.in_ch // 4), self.num_experts, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        b, c, h, w = x.shape
        if c != self.in_ch:
            raise ValueError(f"Expected in_ch={self.in_ch}, got {c}")
        a = self.attn(x).flatten(2).mean(dim=-1)  # (B, K)
        a = torch.softmax(a / max(1e-6, self.temperature), dim=-1)

        w_dyn = torch.einsum("bk,kocij->bocij", a, self.weight)
        b_dyn = torch.einsum("bk,ko->bo", a, self.bias)

        xg = x.reshape(1, b * c, h, w)
        w_g = w_dyn.reshape(b * self.out_ch, self.in_ch, self.k, self.k)
        y = F.conv2d(xg, w_g, bias=None, stride=self.stride, padding=self.padding, groups=b)
        y = y.reshape(b, self.out_ch, y.shape[-2], y.shape[-1])
        y = y + b_dyn[:, :, None, None]
        return y


class DynamicConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int, experts: int) -> None:
        super().__init__()
        self.conv = DynamicConv2d(int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), num_experts=int(experts))
        self.bn = nn.BatchNorm2d(int(out_ch))
        self.act = nn.ReLU(inplace=True)
        self.pw = ConvBNAct(int(out_ch), int(out_ch), kernel_size=1, stride=1, padding=0, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(self.bn(x))
        return self.pw(x)


class DynamicConvNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int, int] = (48, 96, 192, 384),
        experts: int = 4,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        chs = tuple(scale_channels(int(c), float(width_mult), min_ch=16, divisor=8) for c in channels)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), chs[0], kernel_size=3, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = DynamicConvBlock(chs[0], chs[0], stride=1, experts=int(experts))
        self.stage2 = DynamicConvBlock(chs[0], chs[1], stride=2, experts=int(experts))
        self.stage3 = DynamicConvBlock(chs[1], chs[2], stride=2, experts=int(experts))
        self.stage4 = DynamicConvBlock(chs[2], chs[3], stride=2, experts=int(experts))
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
    "dynamic_conv_tiny": {"channels": (40, 80, 160, 320), "experts": 4},
    "dynamic_conv_base": {"channels": (48, 96, 192, 384), "experts": 4},
    "dynamic_conv_wide": {"channels": (64, 128, 256, 512), "experts": 8},
}


def build_dynamic_conv_net_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "dynamic_conv_base",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DynamicConvNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return DynamicConvNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        channels=tuple(map(int, spec["channels"])),
        experts=int(spec["experts"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_dynamic_conv_net_classifier(in_channels=3, num_classes=10, variant="dynamic_conv_tiny", width_mult=0.5)
    y = m(x)
    print("dynamic_conv_tiny", tuple(y.shape))

