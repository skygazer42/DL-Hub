from __future__ import annotations

from torch import nn

from ._common import build_baseline_diffusion_family, smoke_test_diffusion

_VARIANTS: dict[str, dict[str, int]] = {
    "masked_diffusion_tiny": {"width": 64, "depth": 2, "latent": 64},
    "masked_diffusion_small": {"width": 96, "depth": 3, "latent": 96},
    "masked_diffusion_base": {"width": 128, "depth": 4, "latent": 128},
}


def build_masked_diffusion_diffusion(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str = "masked_diffusion_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_baseline_diffusion_family(
        family="masked_diffusion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        image_size=int(image_size),
        latent_dim=int(latent_dim),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
        use_condition=False,
        latent_space=False,
        prediction_mode="x0",
        step_scale=1.0,
    )


if __name__ == "__main__":
    smoke_test_diffusion(build_masked_diffusion_diffusion, "masked_diffusion_tiny")
