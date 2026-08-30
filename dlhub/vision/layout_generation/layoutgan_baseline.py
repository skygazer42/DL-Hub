from __future__ import annotations

from ._common import build_compact_layout_generator, validate_layout_generator

_VARIANTS = {
    "layoutgan_baseline_tiny": {"width": 24, "depth": 1},
    "layoutgan_baseline_small": {"width": 32, "depth": 2},
    "layoutgan_baseline_base": {"width": 48, "depth": 3},
}


def build_layoutgan_baseline_layout_generator(
    *, in_channels: int, variant: str = "layoutgan_baseline_small", width_mult: float = 1.0
):
    return build_compact_layout_generator(
        family="layoutgan_baseline",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    validate_layout_generator(
        build_layoutgan_baseline_layout_generator, "layoutgan_baseline_tiny"
    )
