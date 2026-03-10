from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.ssd import build_ssd_detector as _build_base

_VARIANTS: dict[str, str] = {
    "overfeat_tiny": "ssd_tiny",
    "overfeat_small": "ssd_small",
    "overfeat_base": "ssd_base",
}


def build_overfeat_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "overfeat_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="OverFeat",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(
        label="overfeat_tiny", builder=build_overfeat_detector, variant="overfeat_tiny"
    )
