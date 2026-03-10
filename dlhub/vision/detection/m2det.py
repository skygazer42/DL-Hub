from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.efficientdet import build_efficientdet_detector as _build_base

_VARIANTS: dict[str, str] = {
    "m2det_tiny": "efficientdet_tiny",
    "m2det_small": "efficientdet_small",
    "m2det_base": "efficientdet_base",
}


def build_m2det_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "m2det_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="M2Det",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(label="m2det_tiny", builder=build_m2det_detector, variant="m2det_tiny")
