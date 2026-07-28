from __future__ import annotations

from torch import nn

from ._common import build_baseline_derainer, smoke_test_derainer


_VARIANTS: dict[str, dict[str, int]] = {
    "diffusion_derain_tiny": {"width": 24, "depth": 1, "steps": 2},
    "diffusion_derain_small": {"width": 32, "depth": 2, "steps": 3},
    "diffusion_derain_base": {"width": 48, "depth": 3, "steps": 4},
}


def build_diffusion_derain_derainer(
    *,
    in_channels: int,
    variant: str = "diffusion_derain_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_derainer(
        family="diffusion_derain",
        mode="diffusion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_derainer(build_diffusion_derain_derainer, "diffusion_derain_tiny")
