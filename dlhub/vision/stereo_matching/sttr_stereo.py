from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "sttr_stereo_tiny": {"width": 24, "depth": 1},
    "sttr_stereo_small": {"width": 32, "depth": 2},
    "sttr_stereo_base": {"width": 48, "depth": 3},
}


def build_sttr_stereo_stereo_matcher(
    *, in_channels: int, variant: str = "sttr_stereo_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="sttr_stereo",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_sttr_stereo_stereo_matcher, "sttr_stereo_tiny")
