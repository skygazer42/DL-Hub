from __future__ import annotations

from torch import nn

from ._common import PartFGVCModel, build_fgvc_model, make_fgvc_variants, smoke_test_classifier


class PartRCNN(PartFGVCModel):
    def __init__(self, *, family: str = "part_rcnn", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("part_rcnn", group="part")


def build_part_rcnn_fgvc_classifier(
    *, in_channels: int, num_classes: int, variant: str = "part_rcnn_small", image_size: int = 64, width_mult: float = 1.0, dropout: float = 0.1
) -> nn.Module:
    return build_fgvc_model(PartRCNN, variants=_VARIANTS, in_channels=in_channels, num_classes=num_classes, variant=variant, image_size=image_size, width_mult=width_mult, dropout=dropout, family="part_rcnn")


if __name__ == "__main__":
    smoke_test_classifier(build_part_rcnn_fgvc_classifier, "part_rcnn_tiny")
