from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "token_defect_tiny": {"width": 24, "depth": 1},
    "token_defect_small": {"width": 32, "depth": 2},
    "token_defect_base": {"width": 48, "depth": 3},
}


def build_token_defect_defect_detector(
    *, in_channels: int, variant: str = "token_defect_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="token_defect",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_token_defect_defect_detector, "token_defect_tiny")
