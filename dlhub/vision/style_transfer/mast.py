from __future__ import annotations

import torch
from torch import nn

from ._common import (
    SpatialCrossAttention,
    StyleCodeEncoder,
    TinyDecoder,
    TinyEncoder,
    _default_variants,
)

_VARIANTS: dict[str, dict[str, int]] = _default_variants("mast")


class MASTStyleTransfer(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        width: int,
        depth: int,
        style_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = TinyEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c_feat = int(self.encoder.out_channels)
        self.attn = SpatialCrossAttention(channels=c_feat, temperature=0.7)
        self.style_encoder = StyleCodeEncoder(
            in_channels=int(in_channels),
            width=max(8, int(width) // 2),
            style_dim=int(style_dim),
        )
        self.to_gate = nn.Linear(int(style_dim), c_feat)
        self.decoder = TinyDecoder(
            out_channels=int(in_channels),
            in_channels=c_feat,
            depth=int(depth),
            dropout=float(dropout),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        f_c = self.encoder(content)
        f_s = self.encoder(style)
        style_code = self.style_encoder(style)
        gate = torch.sigmoid(self.to_gate(style_code)).unsqueeze(-1).unsqueeze(-1)
        attended = self.attn(f_c, f_s)
        fused = f_c + gate * attended
        stylized = self.decoder(fused)
        return {"stylized": stylized, "attn_gate": gate.mean()}


def build_mast_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "mast_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    style_dim: int = 64,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MAST variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return MASTStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        style_dim=int(style_dim),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_mast_style_transfer(in_channels=3, variant="mast_tiny", width_mult=0.5)
    out = m(x, s)
    print("mast_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean() + out["attn_gate"]
    loss.backward()
    print("ok")
