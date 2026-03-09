import torch
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.rtdetr import build_rtdetr_detector as _build_base


_VARIANTS: dict[str, str] = {
    "lw_detr_tiny": "rtdetr_tiny",
    "lw_detr_small": "rtdetr_small",
    "lw_detr_base": "rtdetr_base",
}


def build_lw_detr_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "lw_detr_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="LW-DETR",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(label="lw_detr_tiny", builder=build_lw_detr_detector, variant="lw_detr_tiny")
