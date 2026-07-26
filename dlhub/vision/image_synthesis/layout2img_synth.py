from __future__ import annotations
from ._common import build_toy_generator, smoke_test_generator

_VARIANTS = {
    "layout2img_synth_tiny": {"width": 24, "depth": 1},
    "layout2img_synth_small": {"width": 32, "depth": 2},
    "layout2img_synth_base": {"width": 48, "depth": 3},
}


def build_layout2img_synth_generator(
    *, in_channels: int, variant: str = "layout2img_synth_small", width_mult: float = 1.0
):
    return build_toy_generator(
        family="layout2img_synth",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_generator(build_layout2img_synth_generator, "layout2img_synth_tiny")
