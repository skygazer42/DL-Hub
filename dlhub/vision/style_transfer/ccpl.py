from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import SpatialCrossAttention, TinyDecoder, TinyEncoder, _default_variants

_VARIANTS: dict[str, dict[str, int]] = _default_variants("ccpl")


class CCPLStyleTransfer(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = TinyEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c_feat = int(self.encoder.out_channels)
        self.attn = SpatialCrossAttention(channels=c_feat, temperature=0.85)
        self.patch_proj = nn.Conv2d(c_feat, c_feat, kernel_size=1)
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
        attended = self.attn(f_c, f_s)
        fused = self.fuse(torch.cat([f_c, attended], dim=1))
        stylized = self.decoder(fused)
        patch_a = F.normalize(self.patch_proj(fused), dim=1)
        patch_b = F.normalize(self.patch_proj(f_c), dim=1)
        patch_scores = (patch_a * patch_b).sum(dim=1, keepdim=True)
        return {"stylized": stylized, "patch_scores": patch_scores}


def build_ccpl_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "ccpl_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CCPL variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return CCPLStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_ccpl_style_transfer(in_channels=3, variant="ccpl_tiny", width_mult=0.5)
    out = m(x, s)
    print("ccpl_tiny", tuple(out["stylized"].shape), tuple(out["patch_scores"].shape))
    loss = out["stylized"].mean() + out["patch_scores"].mean()
    loss.backward()
    print("ok")
