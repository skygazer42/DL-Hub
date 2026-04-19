from __future__ import annotations

from torch import nn

from ._common import build_toy_hand_segmentor, smoke_test_hand_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "dual_handseg_tiny": {"width": 24, "depth": 1, "classes": 2},
    "dual_handseg_small": {"width": 36, "depth": 2, "classes": 2},
    "dual_handseg_base": {"width": 48, "depth": 3, "classes": 2},
}


def build_dual_handseg_hand_segmentor(
    *,
    in_channels: int,
    variant: str = "dual_handseg_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_hand_segmentor(
        family="dual_handseg",
        mode="dual",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_segmentor(build_dual_handseg_hand_segmentor, "dual_handseg_tiny")
