from __future__ import annotations
from ._common import build_baseline_face_detector, smoke_test_fd

_VARIANTS = {
    "retinaface_tiny": {"width": 24, "depth": 1},
    "retinaface_small": {"width": 32, "depth": 2},
    "retinaface_base": {"width": 48, "depth": 3},
}


def build_retinaface_face_detector(
    *, in_channels: int, variant: str = "retinaface_small", width_mult: float = 1.0
):
    return build_baseline_face_detector(
        family="retinaface",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_fd(build_retinaface_face_detector, "retinaface_tiny")
