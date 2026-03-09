import torch
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.dino import build_dino_detector as _build_base


_VARIANTS: dict[str, str] = {
    "sa_detr_tiny": "dino_tiny",
    "sa_detr_small": "dino_small",
    "sa_detr_base": "dino_base",
}


def build_sa_detr_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "sa_detr_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="SA-DETR",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(label="sa_detr_tiny", builder=build_sa_detr_detector, variant="sa_detr_tiny")
