from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "stereoformer_tiny": {"width": 24, "depth": 1},
    "stereoformer_small": {"width": 32, "depth": 2},
    "stereoformer_base": {"width": 48, "depth": 3},
}


def build_stereoformer_stereo_matcher(
    *, in_channels: int, variant: str = "stereoformer_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="stereoformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_stereoformer_stereo_matcher, "stereoformer_tiny")
