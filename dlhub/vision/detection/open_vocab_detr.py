"""Compatibility entrypoint for the Open-Vocabulary-DETR-labelled baseline."""

from __future__ import annotations

from ._compact_detr import (
    CompactDetrBaseline,
    build_compact_detr_baseline,
    make_detr_baseline_variants,
    smoke_test_detr_builder,
)


class OpenVocabDetrDetector(CompactDetrBaseline):
    """Open-vocabulary label backed by the shared compact DETR baseline."""

    REGISTERED_ALIAS = "open_vocab_detr"


_VARIANTS = make_detr_baseline_variants(OpenVocabDetrDetector.REGISTERED_ALIAS)


def build_open_vocab_detr_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "open_vocab_detr_tiny",
    width_mult: float = 1.0,
) -> OpenVocabDetrDetector:
    return build_compact_detr_baseline(
        detector_type=OpenVocabDetrDetector,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_test_detr_builder(build_open_vocab_detr_detector, "open_vocab_detr_tiny")
