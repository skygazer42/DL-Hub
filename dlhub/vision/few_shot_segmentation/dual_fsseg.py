from __future__ import annotations

from torch import nn

from ._common import build_baseline_few_shot_segmentor, smoke_test_few_shot_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "dual_fsseg_tiny": {"width": 24, "depth": 1, "classes": 2},
    "dual_fsseg_small": {"width": 36, "depth": 2, "classes": 2},
    "dual_fsseg_base": {"width": 48, "depth": 3, "classes": 2},
}


def build_dual_fsseg_few_shot_segmentor(
    *,
    in_channels: int,
    variant: str = "dual_fsseg_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_few_shot_segmentor(
        family="dual_fsseg",
        mode="dual",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_few_shot_segmentor(build_dual_fsseg_few_shot_segmentor, "dual_fsseg_tiny")
