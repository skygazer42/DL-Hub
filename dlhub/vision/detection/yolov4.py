from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.yolov5 import build_yolov5_detector as _build_base

_VARIANTS: dict[str, str] = {
    "yolov4_tiny": "yolov5_tiny",
    "yolov4_small": "yolov5_small",
    "yolov4_base": "yolov5_base",
}


def build_yolov4_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "yolov4_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="YOLOv4",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(
        label="yolov4_tiny", builder=build_yolov4_detector, variant="yolov4_tiny"
    )
