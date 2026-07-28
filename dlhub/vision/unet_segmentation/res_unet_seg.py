from __future__ import annotations

from torch import nn

from ._common import build_baseline_segmentor, smoke_test_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "res_unet_seg_tiny": {"width": 24, "depth": 1, "classes": 2},
    "res_unet_seg_small": {"width": 36, "depth": 2, "classes": 2},
    "res_unet_seg_base": {"width": 48, "depth": 3, "classes": 2},
}


def build_res_unet_seg_segmentor(
    *,
    in_channels: int,
    variant: str = "res_unet_seg_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_segmentor(
        family="res_unet_seg",
        mode="resunet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_segmentor(build_res_unet_seg_segmentor, "res_unet_seg_tiny")
