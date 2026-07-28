from __future__ import annotations

from torch import nn

from ._common import build_baseline_transparent_depth_model, smoke_test_transparent_depth_model


_VARIANTS: dict[str, dict[str, int]] = {
    "geometry_transdepth_tiny": {"width": 24, "depth": 1},
    "geometry_transdepth_small": {"width": 32, "depth": 2},
    "geometry_transdepth_base": {"width": 48, "depth": 3},
}


def build_geometry_transdepth_transparent_depth_model(
    *,
    in_channels: int,
    variant: str = "geometry_transdepth_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_transparent_depth_model(
        family="geometry_transdepth",
        mode="geometry",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_transparent_depth_model(
        build_geometry_transdepth_transparent_depth_model, "geometry_transdepth_tiny"
    )
