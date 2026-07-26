from __future__ import annotations

import torch
from torch import nn

from ._common import TinyDecoder, TinyEncoder, adain

_VARIANTS: dict[str, dict[str, int]] = {
    "photostyle_tiny": {"width": 24, "depth": 2},
    "photostyle_small": {"width": 32, "depth": 3},
    "photostyle_base": {"width": 48, "depth": 4},
}


class PhotostyleStyleTransfer(nn.Module):
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
        t = adain(f_c, f_s)
        y = self.decoder(t)
        return {"stylized": y}


def build_photostyle_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,  # kept for zoo signature consistency
    variant: str = "photostyle_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown AdaIN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return PhotostyleStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_photostyle_style_transfer(in_channels=3, variant="photostyle_tiny", width_mult=0.5)
    out = m(x, s)
    print("photostyle_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean()
    loss.backward()
    print("ok")
