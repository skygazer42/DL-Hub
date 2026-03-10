from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.yolov8 import build_yolov8_detector as _build_base

_VARIANTS: dict[str, str] = {
    "yolo12_tiny": "yolov8_tiny",
    "yolo12_small": "yolov8_small",
    "yolo12_base": "yolov8_base",
}


def build_yolo12_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "yolo12_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="YOLO12",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(
        label="yolo12_tiny", builder=build_yolo12_detector, variant="yolo12_tiny"
    )
