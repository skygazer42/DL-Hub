from __future__ import annotations

from torch import nn

from ._common import ToyGAN, smoke_test_gan

_VARIANTS: dict[str, dict[str, int]] = {
    "stylegan_xl_tiny": {"width": 112, "depth": 4, "latent": 128},
    "stylegan_xl_small": {"width": 144, "depth": 5, "latent": 160},
    "stylegan_xl_base": {"width": 176, "depth": 6, "latent": 192},
}


def build_stylegan_xl_gan(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str = "stylegan_xl_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    latent = max(int(latent_dim), int(cfg["latent"]))
    return ToyGAN(
        family="stylegan_xl",
        in_channels=int(in_channels),
        image_size=int(image_size),
        latent_dim=latent,
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
        num_classes=int(num_classes),
        use_condition=False,
    )


if __name__ == "__main__":
    smoke_test_gan(build_stylegan_xl_gan, "stylegan_xl_tiny")
