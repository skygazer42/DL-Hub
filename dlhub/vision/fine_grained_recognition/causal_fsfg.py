from torch import nn

from ._common import RelationFGVCModel, build_fgvc_model, make_fgvc_variants, smoke_test_classifier


class CausalFSFG(RelationFGVCModel):
    def __init__(self, *, family: str = "causal_fsfg", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("causal_fsfg", group="relation")


def build_causal_fsfg_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "causal_fsfg_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        CausalFSFG,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="causal_fsfg",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_causal_fsfg_fgvc_classifier, "causal_fsfg_tiny")
