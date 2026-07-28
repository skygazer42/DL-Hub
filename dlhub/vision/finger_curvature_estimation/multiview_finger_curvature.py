from __future__ import annotations

from torch import nn

from ._common import build_baseline_hand_regressor, smoke_test_hand_regressor


_VARIANTS: dict[str, dict[str, int]] = {
    "multiview_finger_curvature_tiny": {"width": 24, "depth": 1},
    "multiview_finger_curvature_small": {"width": 36, "depth": 2},
    "multiview_finger_curvature_base": {"width": 48, "depth": 3},
}


def build_multiview_finger_curvature_finger_curvature_estimator(
    *,
    in_channels: int,
    variant: str = "multiview_finger_curvature_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_hand_regressor(
        family="multiview_finger_curvature",
        mode="multiview",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_regressor(
        build_multiview_finger_curvature_finger_curvature_estimator,
        "multiview_finger_curvature_tiny",
    )
