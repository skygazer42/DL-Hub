from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "mambafeat_tiny": {"width": 24, "depth": 1, "embed": 128},
    "mambafeat_small": {"width": 32, "depth": 2, "embed": 160},
    "mambafeat_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_mambafeat_feature_matcher(
    *, in_channels: int, variant: str = "mambafeat_small", width_mult: float = 1.0
):
    return build_toy_model(
        family="mambafeat",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_model(build_mambafeat_feature_matcher, "mambafeat_tiny")
