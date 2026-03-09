
from torch import nn

from ._common import BilinearFGVCModel, build_fgvc_model, make_fgvc_variants, smoke_test_classifier


class HierarchicalBilinear(BilinearFGVCModel):
    def __init__(self, *, family: str = "hierarchical_bilinear", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("hierarchical_bilinear", group="bilinear")


def build_hierarchical_bilinear_fgvc_classifier(
    *, in_channels: int, num_classes: int, variant: str = "hierarchical_bilinear_small", image_size: int = 64, width_mult: float = 1.0, dropout: float = 0.1
) -> nn.Module:
    return build_fgvc_model(HierarchicalBilinear, variants=_VARIANTS, in_channels=in_channels, num_classes=num_classes, variant=variant, image_size=image_size, width_mult=width_mult, dropout=dropout, family="hierarchical_bilinear")


if __name__ == "__main__":
    smoke_test_classifier(build_hierarchical_bilinear_fgvc_classifier, "hierarchical_bilinear_tiny")
