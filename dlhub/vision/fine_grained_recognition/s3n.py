from __future__ import annotations

from torch import nn

from ._common import PartFGVCModel, build_fgvc_model, make_fgvc_variants, smoke_test_classifier


class S3N(PartFGVCModel):
    def __init__(self, *, family: str = "s3n", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("s3n", group="part")


def build_s3n_fgvc_classifier(
    *, in_channels: int, num_classes: int, variant: str = "s3n_small", image_size: int = 64, width_mult: float = 1.0, dropout: float = 0.1
) -> nn.Module:
    return build_fgvc_model(S3N, variants=_VARIANTS, in_channels=in_channels, num_classes=num_classes, variant=variant, image_size=image_size, width_mult=width_mult, dropout=dropout, family="s3n")


if __name__ == "__main__":
    smoke_test_classifier(build_s3n_fgvc_classifier, "s3n_tiny")
