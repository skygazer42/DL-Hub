from __future__ import annotations

from torch import nn

from ._common import build_baseline_hand_regressor, smoke_test_hand_regressor


_VARIANTS: dict[str, dict[str, int]] = {
    "region_palm_orientation_tiny": {"width": 24, "depth": 1},
    "region_palm_orientation_small": {"width": 36, "depth": 2},
    "region_palm_orientation_base": {"width": 48, "depth": 3},
}


def build_region_palm_orientation_palm_orientation_estimator(
    *,
    in_channels: int,
    variant: str = "region_palm_orientation_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_hand_regressor(
        family="region_palm_orientation",
        mode="region",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_regressor(
        build_region_palm_orientation_palm_orientation_estimator, "region_palm_orientation_tiny"
    )
