from __future__ import annotations

import torch
from torch import nn

from ._common import FiLM, ResBlock, StyleCodeEncoder, _conv_norm_act

_VARIANTS: dict[str, dict[str, int]] = {
    "fast_nst_tiny": {"width": 24, "depth": 2},
    "fast_nst_small": {"width": 32, "depth": 3},
    "fast_nst_base": {"width": 48, "depth": 4},
}


class FastNSTStylizer(nn.Module):
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
        c_in = int(in_channels)
        w = int(width)
        d = max(1, int(depth))

        self.style_encoder = StyleCodeEncoder(
            in_channels=c_in, width=max(8, w // 2), style_dim=int(style_dim)
        )

        self.stem = nn.Sequential(
            _conv_norm_act(c_in, w, kernel=7, stride=1, norm="in"),
            _conv_norm_act(w, w, kernel=3, stride=2, norm="in"),
            _conv_norm_act(w, w, kernel=3, stride=2, norm="in"),
        )
        self.blocks = nn.ModuleList([ResBlock(w, dropout=float(dropout)) for _ in range(d)])
        self.mod = nn.ModuleList([FiLM(channels=w, style_dim=int(style_dim)) for _ in range(d)])
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            _conv_norm_act(w, w, kernel=3, stride=1, norm="in"),
            nn.Upsample(scale_factor=2, mode="nearest"),
            _conv_norm_act(w, w // 2, kernel=3, stride=1, norm="in"),
            nn.Conv2d(w // 2, c_in, kernel_size=3, padding=1),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        style_code = self.style_encoder(style)
        x = self.stem(content.to(torch.float32))
        for blk, film in zip(self.blocks, self.mod, strict=True):
            x = film(blk(x), style_code)
        y = self.head(x)
        return {"stylized": y}


def build_fast_nst_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "fast_nst_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    style_dim: int = 64,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FastNST variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return FastNSTStylizer(
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
    m = build_fast_nst_style_transfer(in_channels=3, variant="fast_nst_tiny", width_mult=0.5)
    out = m(x, s)
    print("fast_nst_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean()
    loss.backward()
    print("ok")
