from __future__ import annotations

from torch import nn

from ._common import build_compact_diffusion_family, smoke_test_diffusion

_VARIANTS: dict[str, dict[str, int]] = {
    "rectified_flow_tiny": {"width": 72, "depth": 2, "latent": 64},
    "rectified_flow_small": {"width": 104, "depth": 3, "latent": 96},
    "rectified_flow_base": {"width": 144, "depth": 4, "latent": 128},
}


def build_rectified_flow_diffusion(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str = "rectified_flow_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_compact_diffusion_family(
        family="rectified_flow",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        image_size=int(image_size),
        latent_dim=int(latent_dim),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
        prediction_mode="flow",
        step_scale=0.5,
    )


if __name__ == "__main__":
    smoke_test_diffusion(build_rectified_flow_diffusion, "rectified_flow_tiny")
