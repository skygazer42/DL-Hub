from __future__ import annotations

from ._common import build_compact_layout_generator, validate_layout_generator

_VARIANTS = {
    "constraint_layout_tiny": {"width": 24, "depth": 1},
    "constraint_layout_small": {"width": 32, "depth": 2},
    "constraint_layout_base": {"width": 48, "depth": 3},
}


def build_constraint_layout_layout_generator(
    *, in_channels: int, variant: str = "constraint_layout_small", width_mult: float = 1.0
):
    return build_compact_layout_generator(
        family="constraint_layout",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    validate_layout_generator(
        build_constraint_layout_layout_generator, "constraint_layout_tiny"
    )
