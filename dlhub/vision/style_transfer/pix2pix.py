from __future__ import annotations

import torch
from torch import nn

from ._common import PatchDiscriminator, TinyUNet

_VARIANTS: dict[str, dict[str, int]] = {
    "pix2pix_tiny": {"width": 24, "depth": 2},
    "pix2pix_small": {"width": 32, "depth": 3},
    "pix2pix_base": {"width": 48, "depth": 4},
}


class Pix2PixStyleTransfer(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(in_channels)
        self.generator = TinyUNet(in_channels=c, out_channels=c, width=int(width), depth=int(depth))
        self.discriminator = PatchDiscriminator(
            in_channels=c * 2,
            width=max(8, int(width) // 2),
            depth=max(1, int(depth) - 1),
            dropout=float(dropout),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        fake = self.generator(content)
        fake_logits = self.discriminator(
            torch.cat([content.to(torch.float32), fake.to(torch.float32)], dim=1)
        )
        out: dict[str, torch.Tensor] = {"stylized": fake, "fake_logits": fake_logits}
        if style is not None:
            real_logits = self.discriminator(
                torch.cat([content.to(torch.float32), style.to(torch.float32)], dim=1)
            )
            out["real_logits"] = real_logits
        return out


def build_pix2pix_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "pix2pix_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Pix2Pix variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return Pix2PixStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    y = torch.randn(2, 3, 64, 64)
    m = build_pix2pix_style_transfer(in_channels=3, variant="pix2pix_tiny", width_mult=0.5)
    out = m(x, y)
    print("pix2pix_tiny", tuple(out["stylized"].shape), tuple(out["fake_logits"].shape))
    loss = out["stylized"].mean() + out["fake_logits"].mean()
    loss.backward()
    print("ok")
