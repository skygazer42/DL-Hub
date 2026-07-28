from __future__ import annotations

from torch import nn

from ._common import CompactGAN, smoke_test_gan

_VARIANTS: dict[str, dict[str, int]] = {
    "srgan_tiny": {"width": 80, "depth": 3, "latent": 80},
    "srgan_small": {"width": 112, "depth": 4, "latent": 112},
    "srgan_base": {"width": 144, "depth": 5, "latent": 144},
}


def build_srgan_gan(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str = "srgan_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    latent = max(int(latent_dim), int(cfg["latent"]))
    return CompactGAN(
        family="srgan",
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
    smoke_test_gan(build_srgan_gan, "srgan_tiny")
