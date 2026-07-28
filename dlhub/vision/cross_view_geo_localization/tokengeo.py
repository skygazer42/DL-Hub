from __future__ import annotations
from ._common import build_baseline_cross_view, smoke_test_cv

_VARIANTS = {
    "tokengeo_tiny": {"width": 24, "depth": 1, "embed": 128},
    "tokengeo_small": {"width": 32, "depth": 2, "embed": 160},
    "tokengeo_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_tokengeo_cross_view_localizer(
    *, in_channels: int, variant: str = "tokengeo_small", width_mult: float = 1.0
):
    return build_baseline_cross_view(
        family="tokengeo",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_cv(build_tokengeo_cross_view_localizer, "tokengeo_tiny")
