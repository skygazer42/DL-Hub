from __future__ import annotations

from torch import nn

from ._common import build_baseline_pedestrian_detector, smoke_test_pedestrian_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "scale_pedestrian_tiny": {"width": 24, "depth": 1, "queries": 24},
    "scale_pedestrian_small": {"width": 36, "depth": 2, "queries": 32},
    "scale_pedestrian_base": {"width": 48, "depth": 3, "queries": 48},
}


def build_scale_pedestrian_pedestrian_detector(
    *,
    in_channels: int,
    variant: str = "scale_pedestrian_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_pedestrian_detector(
        family="scale_pedestrian",
        mode="scale",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_pedestrian_detector(
        build_scale_pedestrian_pedestrian_detector, "scale_pedestrian_tiny"
    )
