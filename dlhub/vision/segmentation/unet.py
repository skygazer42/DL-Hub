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


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(int(in_ch), int(out_ch), kernel_size=2, stride=2)
        self.conv = _DoubleConv(int(out_ch) * 2, int(out_ch))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetSegmenter(nn.Module):
    """U-Net semantic segmentation (toy-first).

    Forward: (B, C, H, W) -> logits (B, num_classes, H, W)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        levels: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        base = int(base_channels)
        lv = int(levels)
        if lv < 2:
            raise ValueError(f"levels must be >= 2, got: {levels}")

        self.inc = _DoubleConv(c_in, base)

        downs: list[nn.Module] = []
        ch = base
        for _ in range(lv - 1):
            downs.append(_Down(ch, ch * 2))
            ch *= 2
        self.downs = nn.ModuleList(downs)

        ups: list[nn.Module] = []
        for _ in range(lv - 1):
            ups.append(_Up(ch, ch // 2))
            ch //= 2
        self.ups = nn.ModuleList(ups)

        self.drop = nn.Dropout2d(p=float(dropout))
        self.outc = nn.Conv2d(base, nc, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        skips: list[torch.Tensor] = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)

        x = self.drop(x)
        return self.outc(x)


_VARIANTS: dict[str, dict] = {
    "unet_tiny": {"base_channels": 16, "levels": 3, "dropout": 0.0},
    "unet_small": {"base_channels": 24, "levels": 4, "dropout": 0.0},
    "unet_base": {"base_channels": 32, "levels": 4, "dropout": 0.1},
    "unet_large": {"base_channels": 48, "levels": 4, "dropout": 0.1},
}


def build_unet_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "unet_base",
    dropout: float | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown U-Net variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return UNetSegmenter(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        base_channels=int(spec["base_channels"]),
        levels=int(spec["levels"]),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_unet_segmenter(in_channels=3, num_classes=2, variant="unet_tiny")
    y = m(x)
    print("unet_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

