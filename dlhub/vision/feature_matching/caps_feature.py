from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "caps_feature_tiny": {"width": 24, "depth": 1, "embed": 128},
    "caps_feature_small": {"width": 32, "depth": 2, "embed": 160},
    "caps_feature_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_caps_feature_feature_matcher(
    *, in_channels: int, variant: str = "caps_feature_small", width_mult: float = 1.0
):
    return build_baseline_model(
        family="caps_feature",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_caps_feature_feature_matcher, "caps_feature_tiny")
