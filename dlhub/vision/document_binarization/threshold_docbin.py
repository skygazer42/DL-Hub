from __future__ import annotations

from torch import nn

from ._common import build_toy_doc_binarizer, smoke_test_doc_binarizer


_VARIANTS: dict[str, dict[str, int]] = {
    "threshold_docbin_tiny": {"width": 24, "depth": 1},
    "threshold_docbin_small": {"width": 36, "depth": 2},
    "threshold_docbin_base": {"width": 48, "depth": 3},
}


def build_threshold_docbin_doc_binarizer(
    *, in_channels: int, variant: str = "threshold_docbin_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_doc_binarizer(
        family="threshold_docbin",
        mode="threshold",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_doc_binarizer(build_threshold_docbin_doc_binarizer, "threshold_docbin_tiny")
