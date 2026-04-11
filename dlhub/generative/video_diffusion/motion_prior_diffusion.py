from __future__ import annotations

from torch import nn

from ..diffusion._common import build_toy_diffusion_family, smoke_test_diffusion

_VARIANTS: dict[str, dict[str, int]] = {
    "motion_prior_diffusion_tiny": {"width": 72, "depth": 2, "latent": 128},
    "motion_prior_diffusion_small": {"width": 112, "depth": 3, "latent": 160},
    "motion_prior_diffusion_base": {"width": 160, "depth": 4, "latent": 192},
}


def build_motion_prior_diffusion_video_diffusion(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str = "motion_prior_diffusion_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_toy_diffusion_family(
        family="motion_prior_diffusion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        image_size=int(image_size),
        latent_dim=int(latent_dim),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
        latent_space=True,
        prediction_mode="score",
    )


if __name__ == "__main__":
    smoke_test_diffusion(
        build_motion_prior_diffusion_video_diffusion,
        "motion_prior_diffusion_tiny",
    )
