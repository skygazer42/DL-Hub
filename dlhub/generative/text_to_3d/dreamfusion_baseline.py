from __future__ import annotations

from ._common import build_baseline_text3d_family, smoke_test_text3d

_VARIANTS: dict[str, dict[str, int]] = {
    "dreamfusion_baseline_tiny": {"width": 64, "depth": 2, "latent": 64},
    "dreamfusion_baseline_small": {"width": 96, "depth": 3, "latent": 96},
    "dreamfusion_baseline_base": {"width": 128, "depth": 4, "latent": 128},
}


def build_dreamfusion_baseline_text3d_generator(
    *,
    in_channels: int,
    latent_dim: int = 64,
    variant: str = "dreamfusion_baseline_tiny",
    width_mult: float = 1.0,
):
    return build_baseline_text3d_family(
        family="dreamfusion_baseline",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        latent_dim=int(latent_dim),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_text3d(build_dreamfusion_baseline_text3d_generator, "dreamfusion_baseline_tiny")
