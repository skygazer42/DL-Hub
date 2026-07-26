from __future__ import annotations

from torch import nn

from ._common import build_toy_pedestrian_detector, smoke_test_pedestrian_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "center_pedestrian_tiny": {"width": 24, "depth": 1, "queries": 24},
    "center_pedestrian_small": {"width": 36, "depth": 2, "queries": 32},
    "center_pedestrian_base": {"width": 48, "depth": 3, "queries": 48},
}


def build_center_pedestrian_pedestrian_detector(
    *,
    in_channels: int,
    variant: str = "center_pedestrian_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_pedestrian_detector(
        family="center_pedestrian",
        mode="center",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_pedestrian_detector(
        build_center_pedestrian_pedestrian_detector, "center_pedestrian_tiny"
    )
