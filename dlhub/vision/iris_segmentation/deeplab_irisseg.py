from __future__ import annotations

from torch import nn

from ._common import build_toy_iris_segmentor, smoke_test_iris_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "deeplab_irisseg_tiny": {"width": 24, "depth": 1},
    "deeplab_irisseg_small": {"width": 36, "depth": 2},
    "deeplab_irisseg_base": {"width": 48, "depth": 3},
}


def build_deeplab_irisseg_iris_segmentor(
    *,
    in_channels: int,
    variant: str = "deeplab_irisseg_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_iris_segmentor(
        family="deeplab_irisseg",
        mode="deeplab",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_iris_segmentor(build_deeplab_irisseg_iris_segmentor, "deeplab_irisseg_tiny")
