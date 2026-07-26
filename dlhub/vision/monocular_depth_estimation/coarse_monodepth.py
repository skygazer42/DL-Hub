from __future__ import annotations

from torch import nn

from ._common import build_toy_depth_model, smoke_test_depth_model


_VARIANTS: dict[str, dict[str, int]] = {
    "coarse_monodepth_tiny": {"width": 24, "depth": 1},
    "coarse_monodepth_small": {"width": 36, "depth": 2},
    "coarse_monodepth_base": {"width": 48, "depth": 3},
}


def build_coarse_monodepth_depth_model(
    *,
    in_channels: int,
    variant: str = "coarse_monodepth_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_depth_model(
        family="coarse_monodepth",
        mode="coarse",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_depth_model(build_coarse_monodepth_depth_model, "coarse_monodepth_tiny")
