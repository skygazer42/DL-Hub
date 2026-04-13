from __future__ import annotations
from torch import nn
from ._common import build_toy_document_vlm, smoke_test_document_vlm

_VARIANTS: dict[str, dict[str, int]] = {
    "docformer_doc_tiny": {"width": 24, "depth": 1},
    "docformer_doc_small": {"width": 32, "depth": 2},
    "docformer_doc_base": {"width": 48, "depth": 3},
}

def build_docformer_doc_document_vlm(*, in_channels: int, variant: str = "docformer_doc_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_document_vlm(family="docformer_doc", mode="docformer", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))

if __name__ == "__main__":
    smoke_test_document_vlm(build_docformer_doc_document_vlm, "docformer_doc_tiny")
