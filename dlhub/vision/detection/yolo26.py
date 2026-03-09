import torch
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.yolov10 import build_yolov10_detector as _build_base


_VARIANTS: dict[str, str] = {
    "yolo26_tiny": "yolov10_tiny",
    "yolo26_small": "yolov10_small",
    "yolo26_base": "yolov10_base",
}


def build_yolo26_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "yolo26_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="YOLO26",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(label="yolo26_tiny", builder=build_yolo26_detector, variant="yolo26_tiny")
