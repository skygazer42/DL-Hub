from __future__ import annotations

from torch import nn

from ._common import build_baseline_diffusion_family, smoke_test_diffusion

_VARIANTS: dict[str, dict[str, int]] = {
    "flow_matching_tiny": {"width": 72, "depth": 2, "latent": 64},
    "flow_matching_small": {"width": 112, "depth": 3, "latent": 96},
    "flow_matching_base": {"width": 152, "depth": 4, "latent": 128},
}


def build_flow_matching_diffusion(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str = "flow_matching_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_baseline_diffusion_family(
        family="flow_matching",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        image_size=int(image_size),
        latent_dim=int(latent_dim),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
        prediction_mode="flow",
        step_scale=0.75,
    )


if __name__ == "__main__":
    smoke_test_diffusion(build_flow_matching_diffusion, "flow_matching_tiny")
