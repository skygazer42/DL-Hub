from __future__ import annotations
from ._common import build_toy_model, smoke_test_model

_VARIANTS = {
    "openmask_seg_tiny": {"width": 24, "depth": 1},
    "openmask_seg_small": {"width": 32, "depth": 2},
    "openmask_seg_base": {"width": 48, "depth": 3},
}


def build_openmask_seg_open_vocab_segmenter(
    *, in_channels: int, variant: str = "openmask_seg_small", width_mult: float = 1.0, **kwargs
):
    return build_toy_model(
        family="openmask_seg",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_openmask_seg_open_vocab_segmenter, "openmask_seg_tiny")
