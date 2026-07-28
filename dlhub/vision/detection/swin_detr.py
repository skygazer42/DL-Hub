"""Compatibility entrypoint for the Swin-DETR-labelled baseline."""

from __future__ import annotations

from ._compact_detr import (
    CompactDetrBaseline,
    build_compact_detr_baseline,
    make_detr_baseline_variants,
    smoke_test_detr_builder,
)


class SwinDetrDetector(CompactDetrBaseline):
    """Swin-DETR label backed by the shared compact DETR baseline."""

    REGISTERED_ALIAS = "swin_detr"


_VARIANTS = make_detr_baseline_variants(SwinDetrDetector.REGISTERED_ALIAS)


def build_swin_detr_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "swin_detr_tiny",
    width_mult: float = 1.0,
) -> SwinDetrDetector:
    return build_compact_detr_baseline(
        detector_type=SwinDetrDetector,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_test_detr_builder(build_swin_detr_detector, "swin_detr_tiny")
