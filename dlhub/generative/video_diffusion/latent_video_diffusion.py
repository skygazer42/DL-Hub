from __future__ import annotations

from torch import nn

from ..diffusion._common import build_baseline_diffusion_family, smoke_test_diffusion

_VARIANTS: dict[str, dict[str, int]] = {
    "latent_video_diffusion_tiny": {"width": 80, "depth": 2, "latent": 128},
    "latent_video_diffusion_small": {"width": 128, "depth": 3, "latent": 160},
    "latent_video_diffusion_base": {"width": 176, "depth": 4, "latent": 192},
}


def build_latent_video_diffusion_video_diffusion(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str = "latent_video_diffusion_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_baseline_diffusion_family(
        family="latent_video_diffusion",
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
    smoke_test_diffusion(
        build_latent_video_diffusion_video_diffusion,
        "latent_video_diffusion_tiny",
    )
