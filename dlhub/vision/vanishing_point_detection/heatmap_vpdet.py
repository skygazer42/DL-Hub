from __future__ import annotations

from torch import nn

from ._common import build_baseline_vp_detector, smoke_test_vp_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "heatmap_vpdet_tiny": {"width": 24, "depth": 1},
    "heatmap_vpdet_small": {"width": 36, "depth": 2},
    "heatmap_vpdet_base": {"width": 48, "depth": 3},
}


def build_heatmap_vpdet_vp_detector(
    *, in_channels: int, variant: str = "heatmap_vpdet_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_vp_detector(
        family="heatmap_vpdet",
        mode="heatmap",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_vp_detector(build_heatmap_vpdet_vp_detector, "heatmap_vpdet_tiny")
