import torch
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.detr import build_detr_detector as _build_base


_VARIANTS: dict[str, str] = {
    "owl_vit_tiny": "detr_tiny",
    "owl_vit_small": "detr_small",
    "owl_vit_base": "detr_base",
}


def build_owl_vit_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "owl_vit_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="OWL-ViT",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(label="owl_vit_tiny", builder=build_owl_vit_detector, variant="owl_vit_tiny")
