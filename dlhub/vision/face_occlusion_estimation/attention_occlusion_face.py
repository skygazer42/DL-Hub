from __future__ import annotations

from torch import nn

from ._common import build_toy_face_occlusion_estimator, smoke_test_face_occlusion_estimator


_VARIANTS: dict[str, dict[str, int]] = {'attention_occlusion_face_tiny': {'width': 24, 'depth': 1}, 'attention_occlusion_face_small': {'width': 36, 'depth': 2}, 'attention_occlusion_face_base': {'width': 48, 'depth': 3}}


def build_attention_occlusion_face_occlusion_estimator(
    *,
    in_channels: int,
    variant: str = 'attention_occlusion_face_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_face_occlusion_estimator(
        family='attention_occlusion_face',
        mode='attention',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_face_occlusion_estimator(build_attention_occlusion_face_occlusion_estimator, 'attention_occlusion_face_tiny')
