from __future__ import annotations

from torch import nn

from ._common import RelationFGVCModel, build_fgvc_model, make_fgvc_variants, smoke_test_classifier


class GACNN(RelationFGVCModel):
    def __init__(self, *, family: str = "ga_cnn", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("ga_cnn", group="relation")


def build_ga_cnn_fgvc_classifier(
    *, in_channels: int, num_classes: int, variant: str = "ga_cnn_small", image_size: int = 64, width_mult: float = 1.0, dropout: float = 0.1
) -> nn.Module:
    return build_fgvc_model(GACNN, variants=_VARIANTS, in_channels=in_channels, num_classes=num_classes, variant=variant, image_size=image_size, width_mult=width_mult, dropout=dropout, family="ga_cnn")


if __name__ == "__main__":
    smoke_test_classifier(build_ga_cnn_fgvc_classifier, "ga_cnn_tiny")
