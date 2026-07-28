from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "r3d_vqa_tiny": {"width": 24, "depth": 1},
    "r3d_vqa_small": {"width": 32, "depth": 2},
    "r3d_vqa_base": {"width": 48, "depth": 3},
}


def build_r3d_vqa_vqa_model(
    *, in_channels: int, variant: str = "r3d_vqa_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="r3d_vqa",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_r3d_vqa_vqa_model, "r3d_vqa_tiny")
