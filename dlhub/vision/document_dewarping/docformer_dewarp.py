from __future__ import annotations

from ._common import build_baseline_vision_direction, smoke_test_direction

_VARIANTS = {
    "docformer_dewarp_tiny": {"width": 24, "depth": 1},
    "docformer_dewarp_small": {"width": 32, "depth": 2},
    "docformer_dewarp_base": {"width": 48, "depth": 3},
}


def build_docformer_dewarp_dewarper(
    *, in_channels: int, variant: str = "docformer_dewarp_small", width_mult: float = 1.0
):
    return build_baseline_vision_direction(
        family="docformer_dewarp",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_docformer_dewarp_dewarper, "docformer_dewarp_tiny")
