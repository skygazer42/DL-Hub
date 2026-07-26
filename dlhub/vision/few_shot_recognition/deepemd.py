from __future__ import annotations
from ._common import build_toy_fewshot, smoke_test_few

_VARIANTS = {
    "deepemd_tiny": {"width": 24, "depth": 1, "embed": 128},
    "deepemd_small": {"width": 32, "depth": 2, "embed": 160},
    "deepemd_base": {"width": 48, "depth": 3, "embed": 192},
}


def build_deepemd_few_shot_classifier(
    *, in_channels: int, variant: str = "deepemd_small", width_mult: float = 1.0
):
    return build_toy_fewshot(
        family="deepemd",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_few(build_deepemd_few_shot_classifier, "deepemd_tiny")
