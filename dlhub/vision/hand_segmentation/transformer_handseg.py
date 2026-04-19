from __future__ import annotations

from torch import nn

from ._common import build_toy_hand_segmentor, smoke_test_hand_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_handseg_tiny": {"width": 24, "depth": 1, "classes": 2},
    "transformer_handseg_small": {"width": 36, "depth": 2, "classes": 2},
    "transformer_handseg_base": {"width": 48, "depth": 3, "classes": 2},
}


def build_transformer_handseg_hand_segmentor(
    *,
    in_channels: int,
    variant: str = "transformer_handseg_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_hand_segmentor(
        family="transformer_handseg",
        mode="transformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_segmentor(
        build_transformer_handseg_hand_segmentor, "transformer_handseg_tiny"
    )
