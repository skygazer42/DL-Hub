from __future__ import annotations
from ._common import build_baseline_model, smoke_test_model

_VARIANTS = {
    "layoutlmv3_doc_tiny": {"width": 24, "depth": 1},
    "layoutlmv3_doc_small": {"width": 32, "depth": 2},
    "layoutlmv3_doc_base": {"width": 48, "depth": 3},
}


def build_layoutlmv3_doc_document_model(
    *, in_channels: int, variant: str = "layoutlmv3_doc_small", width_mult: float = 1.0, **kwargs
):
    return build_baseline_model(
        family="layoutlmv3_doc",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        **kwargs,
    )


if __name__ == "__main__":
    smoke_test_model(build_layoutlmv3_doc_document_model, "layoutlmv3_doc_tiny")
