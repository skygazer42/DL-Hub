from __future__ import annotations
from ._common import build_baseline_video_understander, smoke_test_vu

_VARIANTS = {
    "e3dlstm_tiny": {"width": 24, "depth": 1},
    "e3dlstm_small": {"width": 32, "depth": 2},
    "e3dlstm_base": {"width": 48, "depth": 3},
}


def build_e3dlstm_(
    *,
    in_channels: int,
    num_classes: int = 8,
    variant: str = "e3dlstm_small",
    width_mult: float = 1.0,
):
    return build_baseline_video_understander(
        family="e3dlstm",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_vu(build_e3dlstm_, "e3dlstm_tiny")
