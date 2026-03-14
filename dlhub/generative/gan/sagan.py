from __future__ import annotations

from torch import nn

from ._common import ToyGAN, smoke_test_gan

_VARIANTS: dict[str, dict[str, int]] = {
    "sagan_tiny": {"width": 96, "depth": 3, "latent": 96},
    "sagan_small": {"width": 128, "depth": 4, "latent": 128},
    "sagan_base": {"width": 160, "depth": 5, "latent": 160},
}


def build_sagan_gan(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 10,
    variant: str = "sagan_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    latent = max(int(latent_dim), int(cfg["latent"]))
    return ToyGAN(
        family="sagan",
        in_channels=int(in_channels),
        image_size=int(image_size),
        latent_dim=latent,
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
        num_classes=int(num_classes),
        use_condition=True,
    )


if __name__ == "__main__":
    smoke_test_gan(build_sagan_gan, "sagan_tiny")
