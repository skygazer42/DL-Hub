import torch
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.fcos import build_fcos_detector as _build_base


_VARIANTS: dict[str, str] = {
    "borderdet_tiny": "fcos_tiny",
    "borderdet_small": "fcos_small",
    "borderdet_base": "fcos_base",
}


def build_borderdet_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "borderdet_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="BorderDet",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(label="borderdet_tiny", builder=build_borderdet_detector, variant="borderdet_tiny")
