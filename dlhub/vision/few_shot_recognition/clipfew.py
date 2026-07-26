from __future__ import annotations
from ._common import build_toy_fewshot, smoke_test_few

_VARIANTS = {
    "clipfew_tiny": {"width": 24, "depth": 1, "embed": 128},
    "clipfew_small": {"width": 32, "depth": 2, "embed": 160},
    "clipfew_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_clipfew_few_shot_classifier(
    *, in_channels: int, variant: str = "clipfew_small", width_mult: float = 1.0
):
    return build_toy_fewshot(
        family="clipfew",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_few(build_clipfew_few_shot_classifier, "clipfew_tiny")
