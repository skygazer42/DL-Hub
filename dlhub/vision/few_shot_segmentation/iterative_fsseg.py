from __future__ import annotations

from torch import nn

from ._common import build_toy_few_shot_segmentor, smoke_test_few_shot_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "iterative_fsseg_tiny": {"width": 24, "depth": 1, "classes": 2},
    "iterative_fsseg_small": {"width": 36, "depth": 2, "classes": 2},
    "iterative_fsseg_base": {"width": 48, "depth": 3, "classes": 2},
}


def build_iterative_fsseg_few_shot_segmentor(
    *,
    in_channels: int,
    variant: str = "iterative_fsseg_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_few_shot_segmentor(
        family="iterative_fsseg",
        mode="iterative",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_few_shot_segmentor(build_iterative_fsseg_few_shot_segmentor, "iterative_fsseg_tiny")
