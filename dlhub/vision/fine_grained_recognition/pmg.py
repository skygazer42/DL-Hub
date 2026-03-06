from __future__ import annotations

from torch import nn

from ._common import PartFGVCModel, build_fgvc_model, make_fgvc_variants, smoke_test_classifier


class PMG(PartFGVCModel):
    def __init__(self, *, family: str = "pmg", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("pmg", group="part")


def build_pmg_fgvc_classifier(
    *, in_channels: int, num_classes: int, variant: str = "pmg_small", image_size: int = 64, width_mult: float = 1.0, dropout: float = 0.1
) -> nn.Module:
    return build_fgvc_model(PMG, variants=_VARIANTS, in_channels=in_channels, num_classes=num_classes, variant=variant, image_size=image_size, width_mult=width_mult, dropout=dropout, family="pmg")


if __name__ == "__main__":
    smoke_test_classifier(build_pmg_fgvc_classifier, "pmg_tiny")
