"""JORDER (Joint Rain Detection and Removal) - compact-first implementation.

Reference (original idea):
- "Joint Rain Detection and Removal from a Single Image" (CVPR 2017)

This repo keeps it lightweight and offline:
- No pretrained weights, no dataset downloads
- A small U-Net predicts both a rain mask and a rain residual; output is `x - mask * residual`.
"""

import torch
import torch.nn.functional as F
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, depth: int) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        layers: list[nn.Module] = []
        layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(d - 1):
            layers.append(nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JORDER(nn.Module):
    """Compact JORDER-style derainer.

    Outputs a denoised/derained image tensor with the same shape as input.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int = 24,
        depth: int = 1,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        w0 = int(width)
        d = int(depth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w0 < 8:
            raise ValueError("width must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.enc1 = _ConvBlock(c_in, w0, depth=d)
        self.down1 = nn.Conv2d(w0, w0 * 2, kernel_size=4, stride=2, padding=1, bias=True)
        self.enc2 = _ConvBlock(w0 * 2, w0 * 2, depth=d)

        self.down2 = nn.Conv2d(w0 * 2, w0 * 4, kernel_size=4, stride=2, padding=1, bias=True)
        self.bott = _ConvBlock(w0 * 4, w0 * 4, depth=d)

        self.up2 = nn.Conv2d(w0 * 4, w0 * 2, kernel_size=1, bias=True)
        self.dec2 = _ConvBlock(w0 * 4, w0 * 2, depth=d)

        self.up1 = nn.Conv2d(w0 * 2, w0, kernel_size=1, bias=True)
        self.dec1 = _ConvBlock(w0 * 2, w0, depth=d)

        # Multi-task heads: rain residual (C channels) and rain mask (1 channel).
        self.residual_head = nn.Conv2d(w0, c_in, kernel_size=3, padding=1, bias=True)
        self.mask_head = nn.Conv2d(w0, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        e1 = self.enc1(x)
        e2 = self.enc2(F.relu(self.down1(e1), inplace=True))
        b = self.bott(F.relu(self.down2(e2), inplace=True))

        u2 = F.interpolate(b, scale_factor=2, mode="nearest")
        u2 = self.up2(u2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, scale_factor=2, mode="nearest")
        u1 = self.up1(u1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        residual = self.residual_head(d1)
        mask = torch.sigmoid(self.mask_head(d1))  # (B,1,H,W)
        return x - residual * mask


_VARIANTS: dict[str, dict] = {
    "jorder_tiny": {"width": 16, "depth": 1},
    "jorder_small": {"width": 24, "depth": 1},
    "jorder_base": {"width": 32, "depth": 2},
}


def build_jorder_denoiser(
    *,
    in_channels: int,
    variant: str = "jorder_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown JORDER variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return JORDER(in_channels=int(in_channels), width=int(spec["width"]), depth=int(spec["depth"]))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_jorder_denoiser(in_channels=1, variant="jorder_tiny")
    y = m(noisy)
    print("jorder_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")
