"""Compatibility entrypoint for the Focal-DETR-labelled baseline."""

from __future__ import annotations

from ._compact_detr import (
    CompactDetrBaseline,
    build_compact_detr_baseline,
    make_detr_baseline_variants,
    smoke_test_detr_builder,
)


class FocalDetrDetector(CompactDetrBaseline):
    """Focal-DETR label backed by the shared compact DETR baseline."""

    REGISTERED_ALIAS = "focal_detr"


_VARIANTS = make_detr_baseline_variants(FocalDetrDetector.REGISTERED_ALIAS)


def build_focal_detr_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "focal_detr_tiny",
    width_mult: float = 1.0,
) -> FocalDetrDetector:
    return build_compact_detr_baseline(
        detector_type=FocalDetrDetector,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_test_detr_builder(build_focal_detr_detector, "focal_detr_tiny")
