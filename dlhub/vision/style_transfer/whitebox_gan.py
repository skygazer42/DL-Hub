from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import PatchDiscriminator, TinyResNetGenerator, _default_variants

_VARIANTS: dict[str, dict[str, int]] = _default_variants("whitebox_gan")


class WhiteBoxGANStyleTransfer(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(in_channels)
        self.generator = TinyResNetGenerator(
            in_channels=c,
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.discriminator = PatchDiscriminator(
            in_channels=c,
            width=max(8, int(width) // 2),
            depth=max(1, int(depth) - 1),
            dropout=float(dropout),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        surface = F.avg_pool2d(content.to(torch.float32), kernel_size=7, stride=1, padding=3)
        edges = content.to(torch.float32) - F.avg_pool2d(
            content.to(torch.float32), kernel_size=3, stride=1, padding=1
        )
        style_bias = style.to(torch.float32).mean(dim=(2, 3), keepdim=True)
        raw = self.generator(content)
        stylized = torch.tanh(raw + 0.2 * surface + 0.1 * style_bias - 0.05 * edges)
        return {
            "stylized": stylized,
            "surface": surface,
            "edges": edges,
            "fake_logits": self.discriminator(stylized.detach()),
            "real_logits": self.discriminator(style.detach()),
        }


def build_whitebox_gan_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "whitebox_gan_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown WhiteBoxGAN variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return WhiteBoxGANStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_whitebox_gan_style_transfer(
        in_channels=3,
        variant="whitebox_gan_tiny",
        width_mult=0.5,
    )
    out = m(x, s)
    print("whitebox_gan_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean() + out["fake_logits"].mean()
    loss.backward()
    print("ok")
