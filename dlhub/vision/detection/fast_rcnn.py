from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.rfcn import build_rfcn_detector as _build_base

_VARIANTS: dict[str, str] = {
    "fast_rcnn_tiny": "rfcn_tiny",
    "fast_rcnn_small": "rfcn_small",
    "fast_rcnn_base": "rfcn_base",
}


def build_fast_rcnn_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "fast_rcnn_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="Fast R-CNN",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(
        label="fast_rcnn_tiny",
        builder=build_fast_rcnn_detector,
        variant="fast_rcnn_tiny",
    )
