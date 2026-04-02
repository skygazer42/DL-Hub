from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TinyDecoder, TinyEncoder, _default_variants, total_variation, wct

_VARIANTS: dict[str, dict[str, int]] = _default_variants("photo_wct")


class PhotoWCTStyleTransfer(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = TinyEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.decoder = TinyDecoder(
            out_channels=int(in_channels),
            in_channels=int(self.encoder.out_channels),
            depth=int(depth),
            dropout=float(dropout),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        f_c = self.encoder(content)
        f_s = self.encoder(style)
        transformed = wct(f_c, f_s)
        preserved = F.avg_pool2d(f_c, kernel_size=3, stride=1, padding=1)
        stylized = self.decoder(0.8 * transformed + 0.2 * preserved)
        return {"stylized": stylized, "photo_smoothness": total_variation(stylized)}


def build_photo_wct_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "photo_wct_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown PhotoWCT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return PhotoWCTStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_photo_wct_style_transfer(in_channels=3, variant="photo_wct_tiny", width_mult=0.5)
    out = m(x, s)
    print("photo_wct_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean() + out["photo_smoothness"]
    loss.backward()
    print("ok")
