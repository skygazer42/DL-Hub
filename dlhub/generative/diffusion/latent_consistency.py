from __future__ import annotations

from torch import nn

from ._common import build_toy_diffusion_family, smoke_test_diffusion

_VARIANTS: dict[str, dict[str, int]] = {
    "latent_consistency_tiny": {"width": 64, "depth": 2, "latent": 64},
    "latent_consistency_small": {"width": 96, "depth": 3, "latent": 96},
    "latent_consistency_base": {"width": 128, "depth": 4, "latent": 128},
}


def build_latent_consistency_diffusion(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str = "latent_consistency_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_toy_diffusion_family(
        family="latent_consistency",
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
    smoke_test_diffusion(build_latent_consistency_diffusion, "latent_consistency_tiny")

