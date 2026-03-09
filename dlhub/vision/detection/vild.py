import torch
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.rtdetr import build_rtdetr_detector as _build_base


_VARIANTS: dict[str, str] = {
    "vild_tiny": "rtdetr_tiny",
    "vild_small": "rtdetr_small",
    "vild_base": "rtdetr_base",
}


def build_vild_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "vild_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="ViLD",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(label="vild_tiny", builder=build_vild_detector, variant="vild_tiny")
