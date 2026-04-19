from __future__ import annotations

from torch import nn

from ._common import build_toy_camera_calibrator, smoke_test_camera_calibrator


_VARIANTS: dict[str, dict[str, int]] = {
    "selfsup_camcal_tiny": {"width": 24, "depth": 1},
    "selfsup_camcal_small": {"width": 36, "depth": 2},
    "selfsup_camcal_base": {"width": 48, "depth": 3},
}


def build_selfsup_camcal_camera_calibrator(
    *,
    in_channels: int,
    variant: str = "selfsup_camcal_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_camera_calibrator(
        family="selfsup_camcal",
        mode="selfsup",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_camera_calibrator(build_selfsup_camcal_camera_calibrator, "selfsup_camcal_tiny")

