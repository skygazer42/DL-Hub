from torch import nn

from ._common import RelationFGVCModel, build_fgvc_model, make_fgvc_variants, smoke_test_classifier


class ProtoPNetFGVC(RelationFGVCModel):
    def __init__(self, *, family: str = "proto_pnet", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("proto_pnet", group="relation")


def build_proto_pnet_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "proto_pnet_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        ProtoPNetFGVC,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="proto_pnet",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_proto_pnet_fgvc_classifier, "proto_pnet_tiny")
