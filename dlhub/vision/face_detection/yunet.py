from __future__ import annotations
from ._common import build_toy_face_detector, smoke_test_fd

_VARIANTS = {
    "yunet_tiny": {"width": 24, "depth": 1},
    "yunet_small": {"width": 32, "depth": 2},
    "yunet_base": {"width": 48, "depth": 3},
}


def build_yunet_face_detector(
    *, in_channels: int, variant: str = "yunet_small", width_mult: float = 1.0
):
    return build_toy_face_detector(
        family="yunet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_fd(build_yunet_face_detector, "yunet_tiny")
