from __future__ import annotations

import torch
from torch import nn

from ._common import StyleCodeEncoder, TinyDecoder, TinyEncoder

_VARIANTS: dict[str, dict[str, int]] = {
    "munit_tiny": {"width": 24, "depth": 2},
    "munit_small": {"width": 32, "depth": 3},
    "munit_base": {"width": 48, "depth": 4},
}


class MUNITStyleTransfer(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, style_dim: int = 64) -> None:
        super().__init__()
        c = int(in_channels)
        w = int(width)
        d = max(1, int(depth))
        self.enc_depth = max(1, d - 1)
        self.content_encoder = TinyEncoder(
            in_channels=c, width=w, depth=self.enc_depth, dropout=0.0
        )
        self.style_encoder = StyleCodeEncoder(
            in_channels=c, width=max(8, w // 2), style_dim=int(style_dim)
        )
        self.to_style = nn.Linear(int(style_dim), int(self.content_encoder.out_channels))
        self.decoder = TinyDecoder(
            out_channels=c,
            in_channels=int(self.content_encoder.out_channels),
            depth=int(self.enc_depth),
            dropout=0.0,
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        c = self.content_encoder(content)
        s = self.style_encoder(style)
        style_bias = self.to_style(s).view(int(c.shape[0]), -1, 1, 1)
        y = self.decoder(c.to(torch.float32) + style_bias)
        return {"stylized": y, "style_code": s}


def build_munit_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "munit_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,  # unused, kept for zoo signature consistency
    style_dim: int = 64,
) -> nn.Module:
    _ = int(image_size)
    _ = float(dropout)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MUNIT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return MUNITStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        style_dim=int(style_dim),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_munit_style_transfer(in_channels=3, variant="munit_tiny", width_mult=0.5)
    out = m(x, s)
    print("munit_tiny", tuple(out["stylized"].shape), tuple(out["style_code"].shape))
    loss = out["stylized"].mean() + out["style_code"].mean()
    loss.backward()
    print("ok")
