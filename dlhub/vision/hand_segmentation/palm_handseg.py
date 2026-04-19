from __future__ import annotations

from torch import nn

from ._common import build_toy_hand_segmentor, smoke_test_hand_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "palm_handseg_tiny": {"width": 24, "depth": 1, "classes": 2},
    "palm_handseg_small": {"width": 36, "depth": 2, "classes": 2},
    "palm_handseg_base": {"width": 48, "depth": 3, "classes": 2},
}


def build_palm_handseg_hand_segmentor(
    *,
    in_channels: int,
    variant: str = "palm_handseg_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_hand_segmentor(
        family="palm_handseg",
        mode="palm",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_segmentor(build_palm_handseg_hand_segmentor, "palm_handseg_tiny")
