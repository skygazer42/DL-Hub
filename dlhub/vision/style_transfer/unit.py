from __future__ import annotations

import torch
from torch import nn

from ._common import TinyDecoder, TinyEncoder, _default_variants

_VARIANTS: dict[str, dict[str, int]] = _default_variants("unit")


class UNITStyleTransfer(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(in_channels)
        self.enc_a = TinyEncoder(
            in_channels=c,
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.enc_b = TinyEncoder(
            in_channels=c,
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c_feat = int(self.enc_a.out_channels)
        self.shared = nn.Sequential(
            nn.Conv2d(c_feat, c_feat, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_feat, c_feat, kernel_size=1),
        )
        self.dec_a = TinyDecoder(
            out_channels=c,
            in_channels=c_feat,
            depth=int(depth),
            dropout=float(dropout),
        )
        self.dec_b = TinyDecoder(
            out_channels=c,
            in_channels=c_feat,
            depth=int(depth),
            dropout=float(dropout),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        z_a = self.shared(self.enc_a(content))
        z_b = self.shared(self.enc_b(style))
        fake_b = self.dec_b(z_a)
        fake_a = self.dec_a(z_b)
        recon_a = self.dec_a(z_a)
        recon_b = self.dec_b(z_b)
        latent_alignment = (z_a.mean(dim=(2, 3)) - z_b.mean(dim=(2, 3))).square().mean()
        return {
            "stylized": fake_b,
            "fake_a": fake_a,
            "recon_a": recon_a,
            "recon_b": recon_b,
            "latent_alignment": latent_alignment,
        }


def build_unit_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "unit_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown UNIT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return UNITStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_unit_style_transfer(in_channels=3, variant="unit_tiny", width_mult=0.5)
    out = m(x, s)
    print("unit_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean() + out["latent_alignment"]
    loss.backward()
    print("ok")
