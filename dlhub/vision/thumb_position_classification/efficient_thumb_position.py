from __future__ import annotations

from torch import nn

from ._common import build_toy_hand_classifier, smoke_test_hand_classifier


_VARIANTS: dict[str, dict[str, int]] = {
    "efficient_thumb_position_tiny": {"width": 24, "depth": 1, "num_classes": 3},
    "efficient_thumb_position_small": {"width": 36, "depth": 2, "num_classes": 3},
    "efficient_thumb_position_base": {"width": 48, "depth": 3, "num_classes": 3},
}


def build_efficient_thumb_position_thumb_position_classifier(
    *,
    in_channels: int,
    variant: str = "efficient_thumb_position_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_hand_classifier(
        family="efficient_thumb_position",
        mode="efficient",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_classifier(
        build_efficient_thumb_position_thumb_position_classifier, "efficient_thumb_position_tiny"
    )
