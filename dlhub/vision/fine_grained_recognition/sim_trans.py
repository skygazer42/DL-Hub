from torch import nn

from ._common import (
    TransformerFGVCModel,
    build_fgvc_model,
    make_fgvc_variants,
    smoke_test_classifier,
)


class SIMTrans(TransformerFGVCModel):
    def __init__(self, *, family: str = "sim_trans", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("sim_trans", group="transformer")


def build_sim_trans_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "sim_trans_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        SIMTrans,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="sim_trans",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_sim_trans_fgvc_classifier, "sim_trans_tiny")
