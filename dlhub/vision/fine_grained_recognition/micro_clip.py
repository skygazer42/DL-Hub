from torch import nn

from ._common import build_fgvc_model, make_fgvc_variants, smoke_test_classifier
from .fg_clip import FGCLIP


class MicroCLIP(FGCLIP):
    def __init__(self, *, family: str = "micro_clip", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("micro_clip", group="transformer")


def build_micro_clip_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "micro_clip_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        MicroCLIP,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="micro_clip",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_micro_clip_fgvc_classifier, "micro_clip_tiny")
