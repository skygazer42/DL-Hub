import torch
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.reppoints import build_reppoints_detector as _build_base


_VARIANTS: dict[str, str] = {
    "point_linking_network_tiny": "reppoints_tiny",
    "point_linking_network_small": "reppoints_small",
    "point_linking_network_base": "reppoints_base",
}


def build_point_linking_network_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "point_linking_network_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="Point Linking Network",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(
        label="point_linking_network_tiny",
        builder=build_point_linking_network_detector,
        variant="point_linking_network_tiny",
    )
