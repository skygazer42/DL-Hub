from __future__ import annotations

from ._common import build_baseline_text3d_family, smoke_test_text3d

_VARIANTS: dict[str, dict[str, int]] = {
    "score_distill_3d_tiny": {"width": 72, "depth": 2, "latent": 80},
    "score_distill_3d_small": {"width": 104, "depth": 3, "latent": 112},
    "score_distill_3d_base": {"width": 136, "depth": 4, "latent": 144},
}


def build_score_distill_3d_text3d_generator(
    *,
    in_channels: int,
    latent_dim: int = 64,
    variant: str = "score_distill_3d_tiny",
    width_mult: float = 1.0,
):
    return build_baseline_text3d_family(
        family="score_distill_3d",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        latent_dim=int(latent_dim),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_text3d(build_score_distill_3d_text3d_generator, "score_distill_3d_tiny")
