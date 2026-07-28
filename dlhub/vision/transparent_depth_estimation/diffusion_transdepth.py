from __future__ import annotations

from torch import nn

from ._common import build_baseline_transparent_depth_model, smoke_test_transparent_depth_model


_VARIANTS: dict[str, dict[str, int]] = {
    "diffusion_transdepth_tiny": {"width": 24, "depth": 1},
    "diffusion_transdepth_small": {"width": 32, "depth": 2},
    "diffusion_transdepth_base": {"width": 48, "depth": 3},
}


def build_diffusion_transdepth_transparent_depth_model(
    *,
    in_channels: int,
    variant: str = "diffusion_transdepth_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_transparent_depth_model(
        family="diffusion_transdepth",
        mode="diffusion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_transparent_depth_model(
        build_diffusion_transdepth_transparent_depth_model, "diffusion_transdepth_tiny"
    )
