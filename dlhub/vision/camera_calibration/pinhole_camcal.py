from __future__ import annotations

from torch import nn

from ._common import build_toy_camera_calibrator, smoke_test_camera_calibrator


_VARIANTS: dict[str, dict[str, int]] = {
    "pinhole_camcal_tiny": {"width": 24, "depth": 1},
    "pinhole_camcal_small": {"width": 36, "depth": 2},
    "pinhole_camcal_base": {"width": 48, "depth": 3},
}


def build_pinhole_camcal_camera_calibrator(
    *,
    in_channels: int,
    variant: str = "pinhole_camcal_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_camera_calibrator(
        family="pinhole_camcal",
        mode="pinhole",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_camera_calibrator(build_pinhole_camcal_camera_calibrator, "pinhole_camcal_tiny")
