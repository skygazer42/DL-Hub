from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.fcos import build_fcos_detector as _build_base

_VARIANTS: dict[str, str] = {
    "pedestrian_fcos": "fcos_tiny",
}


def build_pedestrian_fcos_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "pedestrian_fcos",
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
        label="pedestrian_fcos",
        builder=build_pedestrian_fcos_detector,
        variant="pedestrian_fcos",
    )

