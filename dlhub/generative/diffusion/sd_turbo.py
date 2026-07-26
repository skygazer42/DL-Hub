from __future__ import annotations

from torch import nn

from ._common import build_toy_diffusion_family, smoke_test_diffusion

_VARIANTS: dict[str, dict[str, int]] = {
    "sd_turbo_tiny": {"width": 64, "depth": 2, "latent": 64},
    "sd_turbo_small": {"width": 96, "depth": 3, "latent": 96},
    "sd_turbo_base": {"width": 128, "depth": 4, "latent": 128},
}


def build_sd_turbo_diffusion(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str = "sd_turbo_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_toy_diffusion_family(
        family="sd_turbo",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        image_size=int(image_size),
        latent_dim=int(latent_dim),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
        prediction_mode="eps",
    )


if __name__ == "__main__":
    smoke_test_diffusion(build_sd_turbo_diffusion, "sd_turbo_tiny")
