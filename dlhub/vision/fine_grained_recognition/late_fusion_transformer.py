from torch import nn

from ._common import TransformerFGVCModel, build_fgvc_model, make_fgvc_variants, smoke_test_classifier


class LateFusionTransformer(TransformerFGVCModel):
    def __init__(self, *, family: str = "late_fusion_transformer", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("late_fusion_transformer", group="transformer")


def build_late_fusion_transformer_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "late_fusion_transformer_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        LateFusionTransformer,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="late_fusion_transformer",
    )


if __name__ == "__main__":
    smoke_test_classifier(
        build_late_fusion_transformer_fgvc_classifier,
        "late_fusion_transformer_tiny",
    )
