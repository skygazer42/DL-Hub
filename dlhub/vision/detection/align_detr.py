"""Compatibility entrypoint for the Align-DETR-labelled baseline."""

from __future__ import annotations

from ._compact_detr import (
    CompactDetrBaseline,
    build_compact_detr_baseline,
    make_detr_baseline_variants,
    smoke_test_detr_builder,
)


class AlignDetrDetector(CompactDetrBaseline):
    """Align-DETR label backed by the shared compact DETR baseline."""

    REGISTERED_ALIAS = "align_detr"


_VARIANTS = make_detr_baseline_variants(AlignDetrDetector.REGISTERED_ALIAS)


def build_align_detr_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "align_detr_tiny",
    width_mult: float = 1.0,
) -> AlignDetrDetector:
    return build_compact_detr_baseline(
        detector_type=AlignDetrDetector,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_test_detr_builder(build_align_detr_detector, "align_detr_tiny")
