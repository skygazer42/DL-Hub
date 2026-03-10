from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.deformable_detr import build_deformable_detr_detector as _build_base

_VARIANTS: dict[str, str] = {
    "deta_tiny": "deformable_detr_tiny",
    "deta_small": "deformable_detr_small",
    "deta_base": "deformable_detr_base",
}


def build_deta_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "deta_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="DETA",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(label="deta_tiny", builder=build_deta_detector, variant="deta_tiny")
