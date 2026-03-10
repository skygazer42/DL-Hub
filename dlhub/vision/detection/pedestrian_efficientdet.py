from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.efficientdet import build_efficientdet_detector as _build_base

_VARIANTS: dict[str, str] = {
    "pedestrian_efficientdet": "efficientdet_tiny",
}


def build_pedestrian_efficientdet_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "pedestrian_efficientdet",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="Pedestrian presets",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(
        label="pedestrian_efficientdet",
        builder=build_pedestrian_efficientdet_detector,
        variant="pedestrian_efficientdet",
    )

