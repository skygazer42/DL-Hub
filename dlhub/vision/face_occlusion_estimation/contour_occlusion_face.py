from __future__ import annotations

from torch import nn

from ._common import build_baseline_face_occlusion_estimator, smoke_test_face_occlusion_estimator


_VARIANTS: dict[str, dict[str, int]] = {
    "contour_occlusion_face_tiny": {"width": 24, "depth": 1},
    "contour_occlusion_face_small": {"width": 36, "depth": 2},
    "contour_occlusion_face_base": {"width": 48, "depth": 3},
}


def build_contour_occlusion_face_occlusion_estimator(
    *,
    in_channels: int,
    variant: str = "contour_occlusion_face_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_face_occlusion_estimator(
        family="contour_occlusion_face",
        mode="contour",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_face_occlusion_estimator(
        build_contour_occlusion_face_occlusion_estimator, "contour_occlusion_face_tiny"
    )
