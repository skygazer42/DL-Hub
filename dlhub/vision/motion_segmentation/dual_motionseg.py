from __future__ import annotations

from torch import nn

from ._common import build_baseline_motion_segmentor, smoke_test_motion_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "dual_motionseg_tiny": {"width": 24, "depth": 1},
    "dual_motionseg_small": {"width": 36, "depth": 2},
    "dual_motionseg_base": {"width": 48, "depth": 3},
}


def build_dual_motionseg_motion_segmentor(
    *, in_channels: int, variant: str = "dual_motionseg_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_motion_segmentor(
        family="dual_motionseg",
        mode="dual",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_motion_segmentor(build_dual_motionseg_motion_segmentor, "dual_motionseg_tiny")
