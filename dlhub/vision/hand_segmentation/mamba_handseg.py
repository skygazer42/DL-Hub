from __future__ import annotations

from torch import nn

from ._common import build_baseline_hand_segmentor, smoke_test_hand_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_handseg_tiny": {"width": 24, "depth": 1, "classes": 2},
    "mamba_handseg_small": {"width": 36, "depth": 2, "classes": 2},
    "mamba_handseg_base": {"width": 48, "depth": 3, "classes": 2},
}


def build_mamba_handseg_hand_segmentor(
    *,
    in_channels: int,
    variant: str = "mamba_handseg_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_hand_segmentor(
        family="mamba_handseg",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_hand_segmentor(build_mamba_handseg_hand_segmentor, "mamba_handseg_tiny")
