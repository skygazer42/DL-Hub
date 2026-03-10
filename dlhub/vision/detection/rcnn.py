from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.faster_rcnn import build_faster_rcnn_detector as _build_base

_VARIANTS: dict[str, str] = {
    "rcnn_tiny": "faster_rcnn_tiny",
    "rcnn_small": "faster_rcnn_small",
    "rcnn_base": "faster_rcnn_base",
}


def build_rcnn_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "rcnn_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="R-CNN",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(label="rcnn_tiny", builder=build_rcnn_detector, variant="rcnn_tiny")
