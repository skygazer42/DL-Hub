from __future__ import annotations

import torch
from torch import nn

from ._common import PatchDiscriminator, TinyResNetGenerator, _default_variants

_VARIANTS: dict[str, dict[str, int]] = _default_variants("dualgan")


class DualGANStyleTransfer(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(in_channels)
        d_depth = max(1, int(depth) - 1)
        d_width = max(8, int(width) // 2)
        self.g_ab = TinyResNetGenerator(
            in_channels=c,
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.g_ba = TinyResNetGenerator(
            in_channels=c,
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.d_a = PatchDiscriminator(
            in_channels=c,
            width=d_width,
            depth=d_depth,
            dropout=float(dropout),
        )
        self.d_b = PatchDiscriminator(
            in_channels=c,
            width=d_width,
            depth=d_depth,
            dropout=float(dropout),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        fake_b = self.g_ab(content)
        fake_a = self.g_ba(style)
        rec_a = self.g_ba(fake_b)
        rec_b = self.g_ab(fake_a)
        return {
            "stylized": fake_b,
            "fake_b": fake_b,
            "fake_a": fake_a,
            "rec_a": rec_a,
            "rec_b": rec_b,
            "critic_a_real": self.d_a(content.detach()),
            "critic_a_fake": self.d_a(fake_a.detach()),
            "critic_b_real": self.d_b(style.detach()),
            "critic_b_fake": self.d_b(fake_b.detach()),
        }


def build_dualgan_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "dualgan_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DualGAN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return DualGANStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_dualgan_style_transfer(in_channels=3, variant="dualgan_tiny", width_mult=0.5)
    out = m(x, s)
    print("dualgan_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean() + out["critic_b_fake"].mean()
    loss.backward()
    print("ok")
