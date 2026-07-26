from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "cfnet_stereo_tiny": {"width": 24, "depth": 1},
    "cfnet_stereo_small": {"width": 32, "depth": 2},
    "cfnet_stereo_base": {"width": 48, "depth": 3},
}


def build_cfnet_stereo_stereo_matcher(
    *, in_channels: int, variant: str = "cfnet_stereo_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="cfnet_stereo",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_cfnet_stereo_stereo_matcher, "cfnet_stereo_tiny")
