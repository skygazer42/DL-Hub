from __future__ import annotations

from ._common import build_compact_layout_generator, validate_layout_generator

_VARIANTS = {
    "bbox_generator_tiny": {"width": 24, "depth": 1},
    "bbox_generator_small": {"width": 32, "depth": 2},
    "bbox_generator_base": {"width": 48, "depth": 3},
}


def build_bbox_generator_layout_generator(
    *, in_channels: int, variant: str = "bbox_generator_small", width_mult: float = 1.0
):
    return build_compact_layout_generator(
        family="bbox_generator",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    validate_layout_generator(build_bbox_generator_layout_generator, "bbox_generator_tiny")
