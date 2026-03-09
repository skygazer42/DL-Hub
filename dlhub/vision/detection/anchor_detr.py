import torch
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.detr import build_detr_detector as _build_base


_VARIANTS: dict[str, str] = {
    "anchor_detr_tiny": "detr_tiny",
    "anchor_detr_small": "detr_small",
    "anchor_detr_base": "detr_base",
}


def build_anchor_detr_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "anchor_detr_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="Anchor DETR",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(
        label="anchor_detr_tiny",
        builder=build_anchor_detr_detector,
        variant="anchor_detr_tiny",
    )
