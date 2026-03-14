from __future__ import annotations

from torch import nn

from ._common import build_toy_diffusion_family, smoke_test_diffusion

_VARIANTS: dict[str, dict[str, int]] = {
    "consistency_model_tiny": {"width": 72, "depth": 2, "latent": 96},
    "consistency_model_small": {"width": 120, "depth": 3, "latent": 128},
    "consistency_model_base": {"width": 168, "depth": 4, "latent": 160},
}


def build_consistency_model_diffusion(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str = "consistency_model_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_toy_diffusion_family(
        family="consistency_model",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        image_size=int(image_size),
        latent_dim=int(latent_dim),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
        latent_space=True,
        prediction_mode="consistency",
    )


if __name__ == "__main__":
    smoke_test_diffusion(build_consistency_model_diffusion, "consistency_model_tiny")
