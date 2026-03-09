from torch import nn

from ._common import PartFGVCModel, build_fgvc_model, make_fgvc_variants, smoke_test_classifier


class PPSSL(PartFGVCModel):
    def __init__(self, *, family: str = "pp_ssl", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("pp_ssl", group="part")


def build_pp_ssl_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "pp_ssl_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        PPSSL,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="pp_ssl",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_pp_ssl_fgvc_classifier, "pp_ssl_tiny")
