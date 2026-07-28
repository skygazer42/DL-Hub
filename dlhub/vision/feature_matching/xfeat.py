from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "xfeat_tiny": {"width": 24, "depth": 1, "embed": 128},
    "xfeat_small": {"width": 32, "depth": 2, "embed": 160},
    "xfeat_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_xfeat_feature_matcher(
    *, in_channels: int, variant: str = "xfeat_small", width_mult: float = 1.0
):
    return build_baseline_model(
        family="xfeat",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_xfeat_feature_matcher, "xfeat_tiny")
