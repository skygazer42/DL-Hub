from __future__ import annotations
from torch import nn
from ._common import build_toy_document_vlm, smoke_test_document_vlm

_VARIANTS: dict[str, dict[str, int]] = {
    "retrieval_doc_tiny": {"width": 24, "depth": 1},
    "retrieval_doc_small": {"width": 32, "depth": 2},
    "retrieval_doc_base": {"width": 48, "depth": 3},
}


def build_retrieval_doc_document_vlm(
    *, in_channels: int, variant: str = "retrieval_doc_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_document_vlm(
        family="retrieval_doc",
        mode="retrieval",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_document_vlm(build_retrieval_doc_document_vlm, "retrieval_doc_tiny")
