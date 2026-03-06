from __future__ import annotations

from torch import nn

from ._common import BilinearFGVCModel, build_fgvc_model, make_fgvc_variants, smoke_test_classifier


class KernelPooling(BilinearFGVCModel):
    def __init__(self, *, family: str = "kernel_pooling", **kwargs) -> None:
        super().__init__(family=family, **kwargs)


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("kernel_pooling", group="bilinear")


def build_kernel_pooling_fgvc_classifier(
    *, in_channels: int, num_classes: int, variant: str = "kernel_pooling_small", image_size: int = 64, width_mult: float = 1.0, dropout: float = 0.1
) -> nn.Module:
    return build_fgvc_model(KernelPooling, variants=_VARIANTS, in_channels=in_channels, num_classes=num_classes, variant=variant, image_size=image_size, width_mult=width_mult, dropout=dropout, family="kernel_pooling")


if __name__ == "__main__":
    smoke_test_classifier(build_kernel_pooling_fgvc_classifier, "kernel_pooling_tiny")
