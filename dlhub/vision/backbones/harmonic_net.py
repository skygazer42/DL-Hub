
import math

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


def _dct_2d_basis(k: int, u: int, v: int) -> torch.Tensor:
    k = int(k)
    u = int(u)
    v = int(v)
    y = torch.arange(k, dtype=torch.float32).view(k, 1)
    x = torch.arange(k, dtype=torch.float32).view(1, k)

    def alpha(n: int) -> float:
        return math.sqrt(1.0 / k) if n == 0 else math.sqrt(2.0 / k)

    basis = alpha(u) * alpha(v) * torch.cos((math.pi * (2 * y + 1) * u) / (2 * k)) * torch.cos(
        (math.pi * (2 * x + 1) * v) / (2 * k)
    )
    return basis


class HarmonicConv2d(nn.Module):
    """Harmonic convolution: fixed DCT basis filters + learned 1x1 mixing."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 7,
        num_basis: int = 16,
        stride: int = 1,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        b = int(num_basis)
        if k <= 0 or k % 2 == 0:
            raise ValueError("kernel_size must be positive odd")
        if b <= 0:
            raise ValueError("num_basis must be > 0")
        # Build a small set of low-frequency DCT bases
        bases: list[torch.Tensor] = []
        max_uv = int(math.ceil(math.sqrt(b)))
        for u in range(max_uv):
            for v in range(max_uv):
                if len(bases) >= b:
                    break
                bases.append(_dct_2d_basis(k, u, v))
            if len(bases) >= b:
                break
        bank = torch.stack(bases, dim=0)  # (B, k, k)
        bank = bank[:, None, :, :]  # (B,1,k,k)
        bank = bank.repeat(int(in_ch), 1, 1, 1)  # (B*in_ch,1,k,k)
        self.register_buffer("weight", bank)
        self.in_ch = int(in_ch)
        self.num_basis = bank.shape[0] // self.in_ch
        self.stride = int(stride)
        self.padding = k // 2
        self.mix = nn.Conv2d(int(in_ch) * self.num_basis, int(out_ch), kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(int(out_ch))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # depthwise apply all bases per channel
        b, c, h, w = x.shape
        if c != self.in_ch:
            raise ValueError(f"Expected in_ch={self.in_ch}, got {c}")
        y = nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding, groups=c)
        y = self.mix(y)
        y = self.bn(y)
        return self.act(y)


class HarmonicNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        depths: tuple[int, int, int, int] = (1, 1, 2, 1),
        kernel_size: int = 7,
        num_basis: int = 16,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        chs = tuple(scale_channels(int(c), float(width_mult)) for c in channels)
        self.stem = nn.Sequential(
            HarmonicConv2d(int(in_channels), chs[0], kernel_size=int(kernel_size), num_basis=int(num_basis), stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        def stage(in_ch: int, out_ch: int, depth: int, *, stride: int) -> nn.Sequential:
            layers: list[nn.Module] = [ConvBNAct(in_ch, out_ch, kernel_size=3, stride=int(stride), act="relu")]
            for _ in range(int(depth) - 1):
                layers.append(ConvBNAct(out_ch, out_ch, kernel_size=3, stride=1, act="relu"))
            return nn.Sequential(*layers)

        self.stage1 = stage(chs[0], chs[0], depths[0], stride=1)
        self.stage2 = stage(chs[0], chs[1], depths[1], stride=2)
        self.stage3 = stage(chs[1], chs[2], depths[2], stride=2)
        self.stage4 = stage(chs[2], chs[3], depths[3], stride=2)
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
    "harmonicnet_tiny": {"channels": (48, 96, 192, 384), "depths": (1, 1, 2, 1), "k": 7, "b": 12},
    "harmonicnet_base": {"channels": (64, 128, 256, 512), "depths": (1, 1, 2, 1), "k": 7, "b": 16},
    "harmonicnet_large": {"channels": (80, 160, 320, 640), "depths": (2, 2, 3, 2), "k": 7, "b": 16},
}


def build_harmonic_net_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "harmonicnet_base",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown HarmonicNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return HarmonicNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        channels=tuple(map(int, spec["channels"])),
        depths=tuple(map(int, spec["depths"])),
        kernel_size=int(spec["k"]),
        num_basis=int(spec["b"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_harmonic_net_classifier(in_channels=3, num_classes=10, variant="harmonicnet_base", width_mult=0.5)
    y = m(x)
    print("harmonicnet_base", tuple(y.shape))

