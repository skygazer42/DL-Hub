from __future__ import annotations

from torch import nn

from ._common import build_toy_hand_classifier, smoke_test_hand_classifier


_VARIANTS: dict[str, dict[str, int]] = {
    "skeleton_thumb_position_tiny": {"width": 24, "depth": 1, "num_classes": 3},
    "skeleton_thumb_position_small": {"width": 36, "depth": 2, "num_classes": 3},
    "skeleton_thumb_position_base": {"width": 48, "depth": 3, "num_classes": 3},
}


def build_skeleton_thumb_position_thumb_position_classifier(
    *,
    in_channels: int,
    variant: str = "skeleton_thumb_position_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_hand_classifier(
        family="skeleton_thumb_position",
        mode="skeleton",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_classifier(
        build_skeleton_thumb_position_thumb_position_classifier, "skeleton_thumb_position_tiny"
    )
