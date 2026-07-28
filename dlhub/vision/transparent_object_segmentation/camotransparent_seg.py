from __future__ import annotations

from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "camotransparent_seg_tiny": {"width": 24, "depth": 1},
    "camotransparent_seg_small": {"width": 32, "depth": 2},
    "camotransparent_seg_base": {"width": 48, "depth": 3},
}


def build_camotransparent_seg_transparent_segmenter(
    *,
    in_channels: int,
    variant: str = "camotransparent_seg_small",
    width_mult: float = 1.0,
    **kwargs,
):
    return build_baseline_model(
        family="camotransparent_seg",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_camotransparent_seg_transparent_segmenter, "camotransparent_seg_tiny")
