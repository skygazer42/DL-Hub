from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "mdt_vqa_tiny": {"width": 24, "depth": 1},
    "mdt_vqa_small": {"width": 32, "depth": 2},
    "mdt_vqa_base": {"width": 48, "depth": 3},
}


def build_mdt_vqa_vqa_model(
    *, in_channels: int, variant: str = "mdt_vqa_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="mdt_vqa",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_mdt_vqa_vqa_model, "mdt_vqa_tiny")
