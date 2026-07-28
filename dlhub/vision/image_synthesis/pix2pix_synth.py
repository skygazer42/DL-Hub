from __future__ import annotations
from ._common import build_baseline_generator, smoke_test_generator

_VARIANTS = {
    "pix2pix_synth_tiny": {"width": 24, "depth": 1},
    "pix2pix_synth_small": {"width": 32, "depth": 2},
    "pix2pix_synth_base": {"width": 48, "depth": 3},
}


def build_pix2pix_synth_generator(
    *, in_channels: int, variant: str = "pix2pix_synth_small", width_mult: float = 1.0
):
    return build_baseline_generator(
        family="pix2pix_synth",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_generator(build_pix2pix_synth_generator, "pix2pix_synth_tiny")
