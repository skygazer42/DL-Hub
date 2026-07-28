from __future__ import annotations

from torch import nn

from ._common import build_baseline_diffusion_family, smoke_test_diffusion

_VARIANTS: dict[str, dict[str, int]] = {
    "latent_diffusion_tiny": {"width": 72, "depth": 2, "latent": 96},
    "latent_diffusion_small": {"width": 112, "depth": 3, "latent": 128},
    "latent_diffusion_base": {"width": 160, "depth": 4, "latent": 160},
}


def build_latent_diffusion_diffusion(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str = "latent_diffusion_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_baseline_diffusion_family(
        family="latent_diffusion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        image_size=int(image_size),
        latent_dim=int(latent_dim),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
        latent_space=True,
        prediction_mode="x0",
    )


if __name__ == "__main__":
    smoke_test_diffusion(build_latent_diffusion_diffusion, "latent_diffusion_tiny")
