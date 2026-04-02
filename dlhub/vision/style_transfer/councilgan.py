from __future__ import annotations

import torch
from torch import nn

from ._common import PatchDiscriminator, TinyResNetGenerator, _default_variants

_VARIANTS: dict[str, dict[str, int]] = _default_variants("councilgan")


class CouncilGANStyleTransfer(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        width: int,
        depth: int,
        num_generators: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c = int(in_channels)
        self.generators = nn.ModuleList(
            [
                TinyResNetGenerator(
                    in_channels=c,
                    width=int(width),
                    depth=int(depth),
                    dropout=float(dropout),
                )
                for _ in range(max(2, int(num_generators)))
            ]
        )
        self.discriminator = PatchDiscriminator(
            in_channels=c,
            width=max(8, int(width) // 2),
            depth=max(1, int(depth) - 1),
            dropout=float(dropout),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        style_bias = style.to(torch.float32).mean(dim=(2, 3), keepdim=True)
        members = [g(content) + 0.05 * style_bias for g in self.generators]
        stacked = torch.stack(members, dim=0)
        stylized = stacked.mean(dim=0)
        council_variance = stacked.var(dim=0, unbiased=False).mean()
        return {
            "stylized": stylized,
            "council_variance": council_variance,
            "fake_logits": self.discriminator(stylized.detach()),
            "real_logits": self.discriminator(style.detach()),
        }


def build_councilgan_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "councilgan_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    num_layers: int = 2,
) -> nn.Module:
    _ = int(image_size)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown CouncilGAN variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return CouncilGANStyleTransfer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        num_generators=max(2, int(num_layers) + 1),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 3, 64, 64)
    m = build_councilgan_style_transfer(in_channels=3, variant="councilgan_tiny", width_mult=0.5)
    out = m(x, s)
    print("councilgan_tiny", tuple(out["stylized"].shape))
    loss = out["stylized"].mean() + out["council_variance"]
    loss.backward()
    print("ok")
