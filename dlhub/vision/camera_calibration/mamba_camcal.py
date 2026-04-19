from __future__ import annotations

from torch import nn

from ._common import build_toy_camera_calibrator, smoke_test_camera_calibrator


_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_camcal_tiny": {"width": 24, "depth": 1},
    "mamba_camcal_small": {"width": 36, "depth": 2},
    "mamba_camcal_base": {"width": 48, "depth": 3},
}


def build_mamba_camcal_camera_calibrator(
    *,
    in_channels: int,
    variant: str = "mamba_camcal_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_camera_calibrator(
        family="mamba_camcal",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_camera_calibrator(build_mamba_camcal_camera_calibrator, "mamba_camcal_tiny")

