from __future__ import annotations
from ._common import build_baseline_aligner, smoke_test_align

_VARIANTS = {
    "fan_tiny": {"width": 24, "depth": 1},
    "fan_small": {"width": 32, "depth": 2},
    "fan_base": {"width": 48, "depth": 3},
}


def build_fan_face_aligner(
    *, in_channels: int, variant: str = "fan_small", width_mult: float = 1.0, num_points: int = 68
):
    return build_baseline_aligner(
        family="fan",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        num_points=int(num_points),
    )


if __name__ == "__main__":
    smoke_test_align(build_fan_face_aligner, "fan_tiny")
