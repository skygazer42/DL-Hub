from __future__ import annotations

from torch import nn

from ._common import build_baseline_diffusion_family, smoke_test_diffusion

_VARIANTS: dict[str, dict[str, int]] = {
    "conditional_flow_matching_tiny": {"width": 80, "depth": 2, "latent": 64},
    "conditional_flow_matching_small": {"width": 120, "depth": 3, "latent": 96},
    "conditional_flow_matching_base": {"width": 160, "depth": 4, "latent": 128},
}


def build_conditional_flow_matching_diffusion(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str = "conditional_flow_matching_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_baseline_diffusion_family(
        family="conditional_flow_matching",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        image_size=int(image_size),
        latent_dim=int(latent_dim),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
        use_condition=True,
        prediction_mode="flow",
        step_scale=0.6,
    )


if __name__ == "__main__":
    smoke_test_diffusion(
        build_conditional_flow_matching_diffusion,
        "conditional_flow_matching_tiny",
    )
