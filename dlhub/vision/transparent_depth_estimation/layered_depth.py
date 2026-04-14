from __future__ import annotations

from torch import nn

from ._common import build_toy_transparent_depth_model, smoke_test_transparent_depth_model


_VARIANTS: dict[str, dict[str, int]] = {
    "layered_depth_tiny": {"width": 24, "depth": 1},
    "layered_depth_small": {"width": 32, "depth": 2},
    "layered_depth_base": {"width": 48, "depth": 3},
}


def build_layered_depth_transparent_depth_model(
    *,
    in_channels: int,
    variant: str = "layered_depth_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_transparent_depth_model(
        family="layered_depth",
        mode="layered",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_transparent_depth_model(
        build_layered_depth_transparent_depth_model, "layered_depth_tiny"
    )
