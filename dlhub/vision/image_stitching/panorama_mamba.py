from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "panorama_mamba_tiny": {"width": 24, "depth": 1},
    "panorama_mamba_small": {"width": 32, "depth": 2},
    "panorama_mamba_base": {"width": 48, "depth": 3},
}


def build_panorama_mamba_stitcher(
    *, in_channels: int, variant: str = "panorama_mamba_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="panorama_mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_panorama_mamba_stitcher, "panorama_mamba_tiny")
