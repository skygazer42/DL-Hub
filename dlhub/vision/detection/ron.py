import torch
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.dssd import build_dssd_detector as _build_base


_VARIANTS: dict[str, str] = {
    "ron_tiny": "dssd_tiny",
    "ron_small": "dssd_small",
    "ron_base": "dssd_base",
}


def build_ron_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "ron_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="RON",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(label="ron_tiny", builder=build_ron_detector, variant="ron_tiny")
