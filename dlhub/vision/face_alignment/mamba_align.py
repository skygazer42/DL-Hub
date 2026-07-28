from __future__ import annotations
from ._common import build_baseline_aligner, smoke_test_align

_VARIANTS = {
    "mamba_align_tiny": {"width": 24, "depth": 1},
    "mamba_align_small": {"width": 32, "depth": 2},
    "mamba_align_base": {"width": 48, "depth": 3},
}


def build_mamba_align_face_aligner(
    *,
    in_channels: int,
    variant: str = "mamba_align_small",
    width_mult: float = 1.0,
    num_points: int = 68,
):
    return build_baseline_aligner(
        family="mamba_align",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        num_points=int(num_points),
    )


if __name__ == "__main__":
    smoke_test_align(build_mamba_align_face_aligner, "mamba_align_tiny")
