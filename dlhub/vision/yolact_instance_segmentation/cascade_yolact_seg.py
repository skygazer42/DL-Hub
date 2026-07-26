from __future__ import annotations

from torch import nn

from ._common import build_toy_instance_segmentor, smoke_test_instance_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "cascade_yolact_seg_tiny": {"width": 24, "depth": 1, "queries": 12, "protos": 6},
    "cascade_yolact_seg_small": {"width": 36, "depth": 2, "queries": 16, "protos": 8},
    "cascade_yolact_seg_base": {"width": 48, "depth": 3, "queries": 20, "protos": 10},
}


def build_cascade_yolact_seg_instance_segmentor(
    *,
    in_channels: int,
    variant: str = "cascade_yolact_seg_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_instance_segmentor(
        family="cascade_yolact_seg",
        mode="cascade",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_instance_segmentor(
        build_cascade_yolact_seg_instance_segmentor, "cascade_yolact_seg_tiny"
    )
