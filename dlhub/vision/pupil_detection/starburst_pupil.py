from __future__ import annotations

from torch import nn

from ._common import build_toy_pupil_detector, smoke_test_pupil_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "starburst_pupil_tiny": {"width": 24, "depth": 1},
    "starburst_pupil_small": {"width": 36, "depth": 2},
    "starburst_pupil_base": {"width": 48, "depth": 3},
}


def build_starburst_pupil_pupil_detector(
    *, in_channels: int, variant: str = "starburst_pupil_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_pupil_detector(
        family="starburst_pupil",
        mode="starburst",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_pupil_detector(build_starburst_pupil_pupil_detector, "starburst_pupil_tiny")
