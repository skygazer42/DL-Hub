from __future__ import annotations
from ._common import build_toy_cross_view, smoke_test_cv

_VARIANTS = {
    "cvmnet2_tiny": {"width": 24, "depth": 1, "embed": 128},
    "cvmnet2_small": {"width": 32, "depth": 2, "embed": 160},
    "cvmnet2_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_cvmnet2_cross_view_localizer(
    *, in_channels: int, variant: str = "cvmnet2_small", width_mult: float = 1.0
):
    return build_toy_cross_view(
        family="cvmnet2",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_cv(build_cvmnet2_cross_view_localizer, "cvmnet2_tiny")
