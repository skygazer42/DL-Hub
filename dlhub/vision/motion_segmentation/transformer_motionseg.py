from __future__ import annotations

from torch import nn

from ._common import build_toy_motion_segmentor, smoke_test_motion_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_motionseg_tiny": {"width": 24, "depth": 1},
    "transformer_motionseg_small": {"width": 36, "depth": 2},
    "transformer_motionseg_base": {"width": 48, "depth": 3},
}


def build_transformer_motionseg_motion_segmentor(
    *, in_channels: int, variant: str = "transformer_motionseg_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_motion_segmentor(
        family="transformer_motionseg",
        mode="transformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_motion_segmentor(
        build_transformer_motionseg_motion_segmentor, "transformer_motionseg_tiny"
    )
