from __future__ import annotations

from ._common import build_baseline_vision_direction, smoke_test_direction

_VARIANTS = {
    "relation_layout_tiny": {"width": 24, "depth": 1},
    "relation_layout_small": {"width": 32, "depth": 2},
    "relation_layout_base": {"width": 48, "depth": 3},
}


def build_relation_layout_layout_generator(
    *, in_channels: int, variant: str = "relation_layout_small", width_mult: float = 1.0
):
    return build_baseline_vision_direction(
        family="relation_layout",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_direction(build_relation_layout_layout_generator, "relation_layout_tiny")
