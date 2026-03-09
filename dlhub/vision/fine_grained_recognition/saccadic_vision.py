from torch import nn

from ._common import PartFGVCModel, build_fgvc_model, make_fgvc_variants, smoke_test_classifier


class SaccadicVision(PartFGVCModel):
    def __init__(self, *, family: str = "saccadic_vision", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("saccadic_vision", group="part")


def build_saccadic_vision_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "saccadic_vision_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        SaccadicVision,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="saccadic_vision",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_saccadic_vision_fgvc_classifier, "saccadic_vision_tiny")
