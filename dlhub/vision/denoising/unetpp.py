from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(out_ch)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_ch), int(out_ch), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(out_ch)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetPlusPlus(nn.Module):
    """U-Net++ (Nested U-Net) denoiser (toy-first, pure torch).

    This is a small, fixed-depth U-Net++ with nested skip connections:
    - 4 encoder stages
    - 3 nested decoder stages (x0_1, x0_2, x0_3)

    Predicts a residual/noise map and returns `x - residual`.
    """

    def __init__(self, *, in_channels: int, base_channels: int = 16) -> None:
        super().__init__()
        c_in = int(in_channels)
        base = int(base_channels)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if base < 8:
            raise ValueError("base_channels must be >= 8")

        b0 = base
        b1 = base * 2
        b2 = base * 4
        b3 = base * 8

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder convs (x0_0 ... x3_0)
        self.conv0_0 = _DoubleConv(c_in, b0)
        self.conv1_0 = _DoubleConv(b0, b1)
        self.conv2_0 = _DoubleConv(b1, b2)
        self.conv3_0 = _DoubleConv(b2, b3)

        # Nested decoder convs
        self.conv0_1 = _DoubleConv(b0 + b1, b0)
        self.conv1_1 = _DoubleConv(b1 + b2, b1)
        self.conv2_1 = _DoubleConv(b2 + b3, b2)

        self.conv0_2 = _DoubleConv(b0 + b0 + b1, b0)
        self.conv1_2 = _DoubleConv(b1 + b1 + b2, b1)

        self.conv0_3 = _DoubleConv(b0 + b0 + b0 + b1, b0)

        self.outc = nn.Conv2d(b0, c_in, kernel_size=1, bias=True)

    def _up_to(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="nearest")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        inp = x

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self._up_to(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up_to(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up_to(x3_0, x2_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up_to(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up_to(x2_1, x1_0)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up_to(x1_2, x0_0)], dim=1))

        residual = self.outc(x0_3)
        return inp - residual


_VARIANTS: dict[str, dict] = {
    "unetpp_tiny": {"base_channels": 12},
    "unetpp_small": {"base_channels": 16},
    "unetpp_base": {"base_channels": 24},
}


def build_unetpp_denoiser(*, in_channels: int, variant: str = "unetpp_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown UNet++ variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return UNetPlusPlus(in_channels=int(in_channels), base_channels=int(spec["base_channels"]))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_unetpp_denoiser(in_channels=1, variant="unetpp_tiny")
    y = m(noisy)
    print("unetpp_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

