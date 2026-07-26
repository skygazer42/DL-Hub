from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "palette_color_tiny": {"width": 24, "depth": 1},
    "palette_color_small": {"width": 32, "depth": 2},
    "palette_color_base": {"width": 48, "depth": 3},
}


def build_palette_color_colorizer(
    *, in_channels: int, variant: str = "palette_color_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="palette_color",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_palette_color_colorizer, "palette_color_tiny")
