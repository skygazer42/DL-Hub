from __future__ import annotations

import math

import torch
from torch import nn

from ._common import TinyDecoder, TinyEncoder, wct

_VARIANTS: dict[str, dict[str, int]] = {
    "avatar_net_tiny": {"width": 24, "depth": 2},
    "avatar_net_small": {"width": 32, "depth": 3},
    "avatar_net_base": {"width": 48, "depth": 4},
}


class FeatureDecoration(nn.Module):
    """Toy 'feature decoration' (WCT + lightweight attention refinement)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")
        self.to_q = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = content_feat.shape
        q = self.to_q(content_feat).flatten(2).transpose(1, 2)  # (B, HWc, C)
        k = self.to_k(style_feat).flatten(2)  # (B, C, HWs)
        v = self.to_v(style_feat).flatten(2).transpose(1, 2)  # (B, HWs, C)
        attn = torch.softmax(torch.bmm(q, k) / math.sqrt(max(1.0, float(c))), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(int(b), int(c), int(h), int(w))
        return self.proj(out)


class AvatarNetStyleTransfer(nn.Module):
    """Avatar-Net style transfer (toy).

    The original method decorates content features with style features across multiple scales.
    Here we keep a simplified version:
    - encode features
    - WCT transform (global)
    - attention-based refinement (local)
    - decode back to image space
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int,
        depth: int,
        dropout: float = 0.0,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        self.encoder = TinyEncoder(
            in_channels=c_in,
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c_feat = int(self.encoder.out_channels)
        self.decor = FeatureDecoration(channels=c_feat)
        self.decoder = TinyDecoder(
            out_channels=c_in,
            in_channels=c_feat,
            depth=int(depth),
            dropout=float(dropout),
        )
        self.alpha = float(alpha)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        f_c = self.encoder(content)
        f_s = self.encoder(style)
        t_wct = wct(f_c, f_s)
        t_local = self.decor(f_c, f_s)
        a = float(self.alpha)
        a = 0.0 if a < 0.0 else 1.0 if a > 1.0 else a
        t = (1.0 - a) * t_wct + a * t_local
        y = self.decoder(t)
        return {"stylized": y}


def build_avatar_net_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "avatar_net_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    alpha: float = 0.5,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Avatar-Net variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return AvatarNetStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
        alpha=float(alpha),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_avatar_net_style_transfer(in_channels=3, variant="avatar_net_tiny", width_mult=0.5)
    out = m(x, s)
    print("avatar_net_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean()
    loss.backward()
    print("ok")

