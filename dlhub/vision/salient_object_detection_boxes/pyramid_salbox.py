from __future__ import annotations

from torch import nn

from ._common import build_toy_box_detector, smoke_test_box_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "pyramid_salbox_tiny": {"width": 24, "depth": 1, "queries": 8},
    "pyramid_salbox_small": {"width": 36, "depth": 2, "queries": 12},
    "pyramid_salbox_base": {"width": 48, "depth": 3, "queries": 16},
}


def build_pyramid_salbox_box_detector(
    *,
    in_channels: int,
    variant: str = "pyramid_salbox_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_box_detector(
        family="pyramid_salbox",
        mode="pyramid",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_box_detector(build_pyramid_salbox_box_detector, "pyramid_salbox_tiny")
