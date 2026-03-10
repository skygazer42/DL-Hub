from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.yolov5 import build_yolov5_detector as _build_base

_VARIANTS: dict[str, str] = {
    "ppyolov2_tiny": "yolov5_tiny",
    "ppyolov2_small": "yolov5_small",
    "ppyolov2_base": "yolov5_base",
}


def build_ppyolov2_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "ppyolov2_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="PP-YOLOv2",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(
        label="ppyolov2_tiny", builder=build_ppyolov2_detector, variant="ppyolov2_tiny"
    )
