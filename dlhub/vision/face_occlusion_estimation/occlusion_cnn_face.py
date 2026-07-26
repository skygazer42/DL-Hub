from __future__ import annotations

from torch import nn

from ._common import build_toy_face_occlusion_estimator, smoke_test_face_occlusion_estimator


_VARIANTS: dict[str, dict[str, int]] = {
    "occlusion_cnn_face_tiny": {"width": 24, "depth": 1},
    "occlusion_cnn_face_small": {"width": 36, "depth": 2},
    "occlusion_cnn_face_base": {"width": 48, "depth": 3},
}


def build_occlusion_cnn_face_occlusion_estimator(
    *,
    in_channels: int,
    variant: str = "occlusion_cnn_face_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_face_occlusion_estimator(
        family="occlusion_cnn_face",
        mode="occlusion_cnn",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_face_occlusion_estimator(
        build_occlusion_cnn_face_occlusion_estimator, "occlusion_cnn_face_tiny"
    )
