"""Compatibility entrypoint for the Prompt-DETR-labelled baseline."""

from __future__ import annotations

from ._compact_detr import (
    CompactDetrBaseline,
    build_compact_detr_baseline,
    make_detr_baseline_variants,
    smoke_test_detr_builder,
)


class PromptDetrDetector(CompactDetrBaseline):
    """Prompt-DETR label backed by the shared compact DETR baseline."""

    REGISTERED_ALIAS = "prompt_detr"


_VARIANTS = make_detr_baseline_variants(PromptDetrDetector.REGISTERED_ALIAS)


def build_prompt_detr_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "prompt_detr_tiny",
    width_mult: float = 1.0,
) -> PromptDetrDetector:
    return build_compact_detr_baseline(
        detector_type=PromptDetrDetector,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_test_detr_builder(build_prompt_detr_detector, "prompt_detr_tiny")
