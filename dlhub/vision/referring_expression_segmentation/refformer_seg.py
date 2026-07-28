from __future__ import annotations
from ._common import build_text_guided_segmentation_baseline, smoke_test_model

_VARIANTS = {
    "refformer_seg_tiny": {"width": 24, "depth": 1},
    "refformer_seg_small": {"width": 32, "depth": 2},
    "refformer_seg_base": {"width": 48, "depth": 3},
}


def build_refformer_seg_refexp_segmenter(
    *, in_channels: int, variant: str = "refformer_seg_small", width_mult: float = 1.0, **kwargs
):
    return build_text_guided_segmentation_baseline(
        registered_alias="refformer_seg",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_refformer_seg_refexp_segmenter, "refformer_seg_tiny")
