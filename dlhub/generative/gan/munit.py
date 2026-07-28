from __future__ import annotations

from torch import nn

from ._common import CompactGAN, smoke_test_gan

_VARIANTS: dict[str, dict[str, int]] = {
    "munit_tiny": {"width": 72, "depth": 3, "latent": 80},
    "munit_small": {"width": 104, "depth": 4, "latent": 112},
    "munit_base": {"width": 136, "depth": 5, "latent": 144},
}


def build_munit_gan(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str = "munit_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    latent = max(int(latent_dim), int(cfg["latent"]))
    return CompactGAN(
        family="munit",
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
    smoke_test_gan(build_munit_gan, "munit_tiny")
