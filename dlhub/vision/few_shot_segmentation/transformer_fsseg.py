from __future__ import annotations

from torch import nn

from ._common import build_toy_few_shot_segmentor, smoke_test_few_shot_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_fsseg_tiny": {"width": 24, "depth": 1, "classes": 2},
    "transformer_fsseg_small": {"width": 36, "depth": 2, "classes": 2},
    "transformer_fsseg_base": {"width": 48, "depth": 3, "classes": 2},
}


def build_transformer_fsseg_few_shot_segmentor(
    *,
    in_channels: int,
    variant: str = "transformer_fsseg_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_few_shot_segmentor(
        family="transformer_fsseg",
        mode="transformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_few_shot_segmentor(
        build_transformer_fsseg_few_shot_segmentor, "transformer_fsseg_tiny"
    )
