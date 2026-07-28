from __future__ import annotations

from torch import nn

from ._common import build_baseline_landmark_detector, smoke_test_landmark_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "graph_face_landmark_tiny": {"width": 24, "depth": 1, "num_landmarks": 16},
    "graph_face_landmark_small": {"width": 36, "depth": 2, "num_landmarks": 16},
    "graph_face_landmark_base": {"width": 48, "depth": 3, "num_landmarks": 16},
}


def build_graph_face_landmark_landmark_detector(
    *,
    in_channels: int,
    variant: str = "graph_face_landmark_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_landmark_detector(
        family="graph_face_landmark",
        mode="graph",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_landmark_detector(
        build_graph_face_landmark_landmark_detector, "graph_face_landmark_tiny"
    )
