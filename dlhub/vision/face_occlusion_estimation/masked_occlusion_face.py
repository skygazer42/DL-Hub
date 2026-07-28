from __future__ import annotations

from torch import nn

from ._common import build_baseline_face_occlusion_estimator, smoke_test_face_occlusion_estimator


_VARIANTS: dict[str, dict[str, int]] = {
    "masked_occlusion_face_tiny": {"width": 24, "depth": 1},
    "masked_occlusion_face_small": {"width": 36, "depth": 2},
    "masked_occlusion_face_base": {"width": 48, "depth": 3},
}


def build_masked_occlusion_face_occlusion_estimator(
    *,
    in_channels: int,
    variant: str = "masked_occlusion_face_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_face_occlusion_estimator(
        family="masked_occlusion_face",
        mode="masked",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_face_occlusion_estimator(
        build_masked_occlusion_face_occlusion_estimator, "masked_occlusion_face_tiny"
    )
