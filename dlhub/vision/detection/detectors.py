import torch
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.cascade_rcnn import build_cascade_rcnn_detector as _build_base


_VARIANTS: dict[str, str] = {
    "detectors_tiny": "cascade_rcnn_tiny",
    "detectors_small": "cascade_rcnn_small",
    "detectors_base": "cascade_rcnn_base",
}


def build_detectors_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "detectors_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="DetectoRS",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(
        label="detectors_tiny",
        builder=build_detectors_detector,
        variant="detectors_tiny",
    )
