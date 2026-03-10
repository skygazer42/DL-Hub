from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.yolov8 import build_yolov8_detector as _build_base

_VARIANTS: dict[str, str] = {
    "yolo11_tiny": "yolov8_tiny",
    "yolo11_small": "yolov8_small",
    "yolo11_base": "yolov8_base",
}


def build_yolo11_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "yolo11_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="YOLO11",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(
        label="yolo11_tiny", builder=build_yolo11_detector, variant="yolo11_tiny"
    )
