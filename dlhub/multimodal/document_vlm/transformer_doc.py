from __future__ import annotations
from torch import nn
from ._common import build_toy_document_vlm, smoke_test_document_vlm

_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_doc_tiny": {"width": 24, "depth": 1},
    "transformer_doc_small": {"width": 32, "depth": 2},
    "transformer_doc_base": {"width": 48, "depth": 3},
}

def build_transformer_doc_document_vlm(*, in_channels: int, variant: str = "transformer_doc_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_document_vlm(family="transformer_doc", mode="transformer", variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))

if __name__ == "__main__":
    smoke_test_document_vlm(build_transformer_doc_document_vlm, "transformer_doc_tiny")
