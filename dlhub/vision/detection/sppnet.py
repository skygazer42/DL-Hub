import torch
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.faster_rcnn import build_faster_rcnn_detector as _build_base


_VARIANTS: dict[str, str] = {
    "sppnet_tiny": "faster_rcnn_tiny",
    "sppnet_small": "faster_rcnn_small",
    "sppnet_base": "faster_rcnn_base",
}


def build_sppnet_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "sppnet_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="SPPNet",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(label="sppnet_tiny", builder=build_sppnet_detector, variant="sppnet_tiny")
