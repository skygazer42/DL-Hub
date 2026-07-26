from __future__ import annotations

from torch import nn

from ._common import build_toy_motion_segmentor, smoke_test_motion_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "flow_motionseg_tiny": {"width": 24, "depth": 1},
    "flow_motionseg_small": {"width": 36, "depth": 2},
    "flow_motionseg_base": {"width": 48, "depth": 3},
}


def build_flow_motionseg_motion_segmentor(
    *, in_channels: int, variant: str = "flow_motionseg_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_motion_segmentor(
        family="flow_motionseg",
        mode="flow",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_motion_segmentor(build_flow_motionseg_motion_segmentor, "flow_motionseg_tiny")
