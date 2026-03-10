from torch import nn

from ._common import (
    TransformerFGVCModel,
    build_fgvc_model,
    make_fgvc_variants,
    smoke_test_classifier,
)


class UniFGVC(TransformerFGVCModel):
    def __init__(self, *, family: str = "unifgvc", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("unifgvc", group="transformer")


def build_unifgvc_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "unifgvc_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        UniFGVC,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="unifgvc",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_unifgvc_fgvc_classifier, "unifgvc_tiny")
