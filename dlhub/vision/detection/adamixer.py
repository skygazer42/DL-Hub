import torch
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.dab_detr import build_dab_detr_detector as _build_base


_VARIANTS: dict[str, str] = {
    "adamixer_tiny": "dab_detr_tiny",
    "adamixer_small": "dab_detr_small",
    "adamixer_base": "dab_detr_base",
}


def build_adamixer_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "adamixer_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="AdaMixer",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(label="adamixer_tiny", builder=build_adamixer_detector, variant="adamixer_tiny")
