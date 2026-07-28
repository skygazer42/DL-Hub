from __future__ import annotations

from ._common import build_baseline_text3d_family, smoke_test_text3d

_VARIANTS: dict[str, dict[str, int]] = {
    "sdf_prompt_3d_tiny": {"width": 64, "depth": 2, "latent": 68},
    "sdf_prompt_3d_small": {"width": 96, "depth": 3, "latent": 100},
    "sdf_prompt_3d_base": {"width": 128, "depth": 4, "latent": 132},
}


def build_sdf_prompt_3d_text3d_generator(
    *,
    in_channels: int,
    latent_dim: int = 64,
    variant: str = "sdf_prompt_3d_tiny",
    width_mult: float = 1.0,
):
    return build_baseline_text3d_family(
        family="sdf_prompt_3d",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        latent_dim=int(latent_dim),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_text3d(build_sdf_prompt_3d_text3d_generator, "sdf_prompt_3d_tiny")
