from __future__ import annotations

from torch import nn

from ._common import build_toy_liveness_detector, smoke_test_liveness_detector


_VARIANTS: dict[str, dict[str, int]] = {'artifact_face_liveness_tiny': {'width': 24, 'depth': 1, 'num_classes': 2}, 'artifact_face_liveness_small': {'width': 36, 'depth': 2, 'num_classes': 2}, 'artifact_face_liveness_base': {'width': 48, 'depth': 3, 'num_classes': 2}}


def build_artifact_face_liveness_liveness_detector(
    *,
    in_channels: int,
    variant: str = 'artifact_face_liveness_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_liveness_detector(
        family='artifact_face_liveness',
        mode='artifact',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_liveness_detector(build_artifact_face_liveness_liveness_detector, 'artifact_face_liveness_tiny')
