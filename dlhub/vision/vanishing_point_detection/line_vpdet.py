from __future__ import annotations

from torch import nn

from ._common import build_baseline_vp_detector, smoke_test_vp_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "line_vpdet_tiny": {"width": 24, "depth": 1},
    "line_vpdet_small": {"width": 36, "depth": 2},
    "line_vpdet_base": {"width": 48, "depth": 3},
}


def build_line_vpdet_vp_detector(
    *, in_channels: int, variant: str = "line_vpdet_small", width_mult: float = 1.0
) -> nn.Module:
    return build_baseline_vp_detector(
        family="line_vpdet",
        mode="line",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_vp_detector(build_line_vpdet_vp_detector, "line_vpdet_tiny")
