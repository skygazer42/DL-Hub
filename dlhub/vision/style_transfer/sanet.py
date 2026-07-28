from __future__ import annotations

import math

import torch
from torch import nn

from ._common import TinyDecoder, TinyEncoder

_VARIANTS: dict[str, dict[str, int]] = {
    "sanet_tiny": {"width": 24, "depth": 2},
    "sanet_small": {"width": 32, "depth": 3},
    "sanet_base": {"width": 48, "depth": 4},
}


class StyleAttention(nn.Module):
    """Lightweight spatial cross-attention between content/style feature maps."""

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
        if content_feat.ndim != 4 or style_feat.ndim != 4:
            raise ValueError(
                f"Expected (B, C, H, W) for content/style features, got {tuple(content_feat.shape)} and {tuple(style_feat.shape)}"
            )
        b, c, h, w = content_feat.shape

        q = self.to_q(content_feat).flatten(2).transpose(1, 2)  # (B, HWc, C)
        k = self.to_k(style_feat).flatten(2)  # (B, C, HWs)
        v = self.to_v(style_feat).flatten(2).transpose(1, 2)  # (B, HWs, C)

        logits = torch.bmm(q, k) / math.sqrt(max(1.0, float(c)))
        attn = torch.softmax(logits, dim=-1)  # (B, HWc, HWs)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(int(b), int(c), int(h), int(w))
        return self.proj(out)


class SANetStyleTransfer(nn.Module):
    """Style-Attentional Networks (SANet) style transfer (compact).

    This is a compact approximation that keeps the key ingredient: content->style spatial attention
    over encoded feature maps, followed by a tiny decoder.
    """

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        c_in = int(in_channels)
        self.encoder = TinyEncoder(
            in_channels=c_in,
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c_feat = int(self.encoder.out_channels)
        self.attn = StyleAttention(channels=c_feat)
        self.fuse = nn.Sequential(
            nn.Conv2d(c_feat * 2, c_feat, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = TinyDecoder(
            out_channels=c_in,
            in_channels=c_feat,
            depth=int(depth),
            dropout=float(dropout),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        f_c = self.encoder(content)
        f_s = self.encoder(style)
        t = self.attn(f_c, f_s)
        y = self.decoder(self.fuse(torch.cat([f_c, t], dim=1)))
        return {"stylized": y, "attn_strength": t.abs().mean()}


def build_sanet_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "sanet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SANet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return SANetStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_sanet_style_transfer(in_channels=3, variant="sanet_tiny", width_mult=0.5, dropout=0.0)
    out = m(x, s)
    print("sanet_tiny", tuple(out["stylized"].shape), float(out["attn_strength"].item()))
    loss = out["stylized"].mean() + out["attn_strength"]
    loss.backward()
    print("ok")
