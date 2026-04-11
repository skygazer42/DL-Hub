from __future__ import annotations

from torch import nn

from ..diffusion._common import build_toy_diffusion_family, smoke_test_diffusion

_VARIANTS: dict[str, dict[str, int]] = {
    "control_video_diffusion_tiny": {"width": 80, "depth": 2, "latent": 96},
    "control_video_diffusion_small": {"width": 120, "depth": 3, "latent": 128},
    "control_video_diffusion_base": {"width": 168, "depth": 4, "latent": 160},
}


def build_control_video_diffusion_video_diffusion(
    *,
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str = "control_video_diffusion_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_toy_diffusion_family(
        family="control_video_diffusion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        image_size=int(image_size),
        latent_dim=int(latent_dim),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
        use_condition=True,
        prediction_mode="eps",
    )


if __name__ == "__main__":
    smoke_test_diffusion(
        build_control_video_diffusion_video_diffusion,
        "control_video_diffusion_tiny",
    )
