from __future__ import annotations

from torch import nn

from ._common import build_toy_salient_detector, smoke_test_salient_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "focal_salient_tiny": {"width": 24, "depth": 1},
    "focal_salient_small": {"width": 36, "depth": 2},
    "focal_salient_base": {"width": 48, "depth": 3},
}


def build_focal_salient_salient_detector(
    *,
    in_channels: int,
    variant: str = "focal_salient_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_salient_detector(
        family="focal_salient",
        mode="focal",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_salient_detector(build_focal_salient_salient_detector, "focal_salient_tiny")
