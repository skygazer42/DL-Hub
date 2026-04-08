from __future__ import annotations

import torch
from torch import nn

from ._common import TinyDecoder, TinyEncoder

_VARIANTS: dict[str, dict[str, int]] = {
    "diffstyle_tiny": {"width": 24, "depth": 2},
    "diffstyle_small": {"width": 32, "depth": 3},
    "diffstyle_base": {"width": 48, "depth": 4},
}


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(dim)
        if d <= 0:
            raise ValueError("dim must be > 0")
        h = int(num_heads)
        if h <= 0:
            raise ValueError("num_heads must be > 0")
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=h, dropout=float(dropout), batch_first=True
        )
        hidden = max(d, int(d * float(mlp_ratio)))
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, d),
        )

    def forward(self, content_tokens: torch.Tensor, style_tokens: torch.Tensor) -> torch.Tensor:
        # content_tokens: (B, Nc, D), style_tokens: (B, Ns, D)
        q = self.norm_q(content_tokens)
        kv = self.norm_kv(style_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = content_tokens + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class StyTr2StyleTransfer(nn.Module):
    """Transformer style transfer (StyTr2-like, toy).

    Keeps the core idea: cross-attention between content and style feature tokens.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int,
        depth: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        self.encoder = TinyEncoder(
            in_channels=c_in,
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        dim = int(self.encoder.out_channels)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(dim=dim, num_heads=4, mlp_ratio=2.0, dropout=float(dropout))
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.decoder = TinyDecoder(
            out_channels=c_in,
            in_channels=dim,
            depth=int(depth),
            dropout=float(dropout),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        f_c = self.encoder(content)
        f_s = self.encoder(style)
        b, c, h, w = f_c.shape
        content_tokens = f_c.flatten(2).transpose(1, 2)  # (B, Nc, C)
        style_tokens = f_s.flatten(2).transpose(1, 2)  # (B, Ns, C)
        x = content_tokens
        for blk in self.blocks:
            x = blk(x, style_tokens)
        feat = x.transpose(1, 2).reshape(int(b), int(c), int(h), int(w))
        y = self.decoder(feat)
        return {"stylized": y}


def build_diffstyle_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "diffstyle_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    num_layers: int = 2,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown StyTr2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return StyTr2StyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        num_layers=int(num_layers),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_diffstyle_style_transfer(in_channels=3, variant="diffstyle_tiny", width_mult=0.5, num_layers=1)
    out = m(x, s)
    print("diffstyle_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean()
    loss.backward()
    print("ok")



