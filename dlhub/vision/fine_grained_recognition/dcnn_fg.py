from torch import nn

from ._common import RelationFGVCModel, build_fgvc_model, make_fgvc_variants, smoke_test_classifier


class DCNNFG(RelationFGVCModel):
    def __init__(self, *, family: str = "dcnn_fg", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("dcnn_fg", group="relation")


def build_dcnn_fg_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "dcnn_fg_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        DCNNFG,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="dcnn_fg",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_dcnn_fg_fgvc_classifier, "dcnn_fg_tiny")
