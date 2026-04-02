from __future__ import annotations

import torch
from torch import nn

from ._common import PatchDiscriminator, ResBlock, _conv_norm_act

_VARIANTS: dict[str, dict[str, int]] = {
    "cyclegan_tiny": {"width": 24, "depth": 2},
    "cyclegan_small": {"width": 32, "depth": 3},
    "cyclegan_base": {"width": 48, "depth": 4},
}


class TinyResNetGenerator(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(in_channels)
        w = int(width)
        d = max(1, int(depth))
        self.stem = nn.Sequential(
            _conv_norm_act(c, w, kernel=7, stride=1, norm="in"),
            _conv_norm_act(w, w * 2, kernel=3, stride=2, norm="in"),
            _conv_norm_act(w * 2, w * 4, kernel=3, stride=2, norm="in"),
        )
        cur = w * 4
        self.blocks = nn.Sequential(*[ResBlock(cur, dropout=float(dropout)) for _ in range(d)])
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            _conv_norm_act(cur, w * 2, kernel=3, stride=1, norm="in"),
            nn.Upsample(scale_factor=2, mode="nearest"),
            _conv_norm_act(w * 2, w, kernel=3, stride=1, norm="in"),
            nn.Conv2d(w, c, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        return self.head(self.blocks(self.stem(x)))


class CycleGANStyleTransfer(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(in_channels)
        self.g_ab = TinyResNetGenerator(
            in_channels=c, width=int(width), depth=int(depth), dropout=float(dropout)
        )
        self.g_ba = TinyResNetGenerator(
            in_channels=c, width=int(width), depth=int(depth), dropout=float(dropout)
        )
        self.d_a = PatchDiscriminator(
            in_channels=c,
            width=max(8, int(width) // 2),
            depth=max(1, int(depth) - 1),
            dropout=float(dropout),
        )
        self.d_b = PatchDiscriminator(
            in_channels=c,
            width=max(8, int(width) // 2),
            depth=max(1, int(depth) - 1),
            dropout=float(dropout),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        a = content
        b = style
        fake_b = self.g_ab(a)
        rec_a = self.g_ba(fake_b)
        fake_a = self.g_ba(b)
        rec_b = self.g_ab(fake_a)

        return {
            "stylized": fake_b,
            "fake_b": fake_b,
            "rec_a": rec_a,
            "fake_a": fake_a,
            "rec_b": rec_b,
            "logits_d_a_fake": self.d_a(fake_a.detach()),
            "logits_d_a_real": self.d_a(a.detach()),
            "logits_d_b_fake": self.d_b(fake_b.detach()),
            "logits_d_b_real": self.d_b(b.detach()),
        }


def build_cyclegan_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "cyclegan_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CycleGAN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return CycleGANStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(1, 3, 64, 64)
    b = torch.randn(1, 3, 64, 64)
    m = build_cyclegan_style_transfer(in_channels=3, variant="cyclegan_tiny", width_mult=0.5)
    out = m(a, b)
    print("cyclegan_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean()
    loss.backward()
    print("ok")
