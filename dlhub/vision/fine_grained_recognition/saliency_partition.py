from torch import nn

from ._common import RelationFGVCModel, build_fgvc_model, make_fgvc_variants, smoke_test_classifier


class SaliencyPartition(RelationFGVCModel):
    def __init__(self, *, family: str = "saliency_partition", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("saliency_partition", group="relation")


def build_saliency_partition_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "saliency_partition_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        SaliencyPartition,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="saliency_partition",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_saliency_partition_fgvc_classifier, "saliency_partition_tiny")
