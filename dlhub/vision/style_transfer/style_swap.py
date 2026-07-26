from __future__ import annotations

import torch
from torch import nn

from ._common import SpatialCrossAttention, TinyDecoder, TinyEncoder, _default_variants

_VARIANTS: dict[str, dict[str, int]] = _default_variants("style_swap")


class StyleSwapStyleTransfer(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = TinyEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c_feat = int(self.encoder.out_channels)
        self.swap = SpatialCrossAttention(channels=c_feat, temperature=0.8)
        self.fuse = nn.Sequential(
            nn.Conv2d(c_feat * 2, c_feat, kernel_size=1), nn.ReLU(inplace=True)
        )
        self.decoder = TinyDecoder(
            out_channels=int(in_channels),
            in_channels=c_feat,
            depth=int(depth),
            dropout=float(dropout),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        f_c = self.encoder(content)
        f_s = self.encoder(style)
        swapped = self.swap(f_c, f_s)
        stylized = self.decoder(self.fuse(torch.cat([f_c, swapped], dim=1)))
        return {"stylized": stylized, "swap_strength": swapped.abs().mean()}


def build_style_swap_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "style_swap_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown StyleSwap variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return StyleSwapStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_style_swap_style_transfer(in_channels=3, variant="style_swap_tiny", width_mult=0.5)
    out = m(x, s)
    print("style_swap_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean() + out["swap_strength"]
    loss.backward()
    print("ok")
