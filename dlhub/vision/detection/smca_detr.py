import torch
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.conditional_detr import build_conditional_detr_detector as _build_base


_VARIANTS: dict[str, str] = {
    "smca_detr_tiny": "conditional_detr_tiny",
    "smca_detr_small": "conditional_detr_small",
    "smca_detr_base": "conditional_detr_base",
}


def build_smca_detr_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "smca_detr_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="SMCA-DETR",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(label="smca_detr_tiny", builder=build_smca_detr_detector, variant="smca_detr_tiny")
