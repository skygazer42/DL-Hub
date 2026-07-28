from __future__ import annotations

from torch import nn

from ._common import build_baseline_hand_classifier, smoke_test_hand_classifier


_VARIANTS: dict[str, dict[str, int]] = {
    "skeleton_gesture_tiny": {"width": 24, "depth": 1, "num_classes": 4},
    "skeleton_gesture_small": {"width": 36, "depth": 2, "num_classes": 4},
    "skeleton_gesture_base": {"width": 48, "depth": 3, "num_classes": 4},
}


def build_skeleton_gesture_gesture_recognizer(
    *,
    in_channels: int,
    variant: str = "skeleton_gesture_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_hand_classifier(
        family="skeleton_gesture",
        mode="skeleton",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_classifier(build_skeleton_gesture_gesture_recognizer, "skeleton_gesture_tiny")
