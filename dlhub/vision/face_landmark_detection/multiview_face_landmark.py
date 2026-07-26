from __future__ import annotations

from torch import nn

from ._common import build_toy_landmark_detector, smoke_test_landmark_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "multiview_face_landmark_tiny": {"width": 24, "depth": 1, "num_landmarks": 16},
    "multiview_face_landmark_small": {"width": 36, "depth": 2, "num_landmarks": 16},
    "multiview_face_landmark_base": {"width": 48, "depth": 3, "num_landmarks": 16},
}


def build_multiview_face_landmark_landmark_detector(
    *,
    in_channels: int,
    variant: str = "multiview_face_landmark_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_landmark_detector(
        family="multiview_face_landmark",
        mode="multiview",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_landmark_detector(
        build_multiview_face_landmark_landmark_detector, "multiview_face_landmark_tiny"
    )
