from __future__ import annotations

from torch import nn

from ._common import build_baseline_hand_regressor, smoke_test_hand_regressor


_VARIANTS: dict[str, dict[str, int]] = {
    "coarse_to_fine_finger_spread_tiny": {"width": 24, "depth": 1},
    "coarse_to_fine_finger_spread_small": {"width": 36, "depth": 2},
    "coarse_to_fine_finger_spread_base": {"width": 48, "depth": 3},
}


def build_coarse_to_fine_finger_spread_finger_spread_estimator(
    *,
    in_channels: int,
    variant: str = "coarse_to_fine_finger_spread_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_hand_regressor(
        family="coarse_to_fine_finger_spread",
        mode="coarse_to_fine",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_regressor(
        build_coarse_to_fine_finger_spread_finger_spread_estimator,
        "coarse_to_fine_finger_spread_tiny",
    )
