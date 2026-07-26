from __future__ import annotations

from torch import nn

from ._common import build_toy_salient_detector, smoke_test_salient_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "diffusion_salient_tiny": {"width": 24, "depth": 1},
    "diffusion_salient_small": {"width": 36, "depth": 2},
    "diffusion_salient_base": {"width": 48, "depth": 3},
}


def build_diffusion_salient_salient_detector(
    *,
    in_channels: int,
    variant: str = "diffusion_salient_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_salient_detector(
        family="diffusion_salient",
        mode="diffusion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_salient_detector(build_diffusion_salient_salient_detector, "diffusion_salient_tiny")
