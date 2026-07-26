from __future__ import annotations
from ._common import build_toy_face_detector, smoke_test_fd

_VARIANTS = {
    "dsfd_tiny": {"width": 24, "depth": 1},
    "dsfd_small": {"width": 32, "depth": 2},
    "dsfd_base": {"width": 48, "depth": 3},
}


def build_dsfd_face_detector(
    *, in_channels: int, variant: str = "dsfd_small", width_mult: float = 1.0
):
    return build_toy_face_detector(
        family="dsfd",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_fd(build_dsfd_face_detector, "dsfd_tiny")
