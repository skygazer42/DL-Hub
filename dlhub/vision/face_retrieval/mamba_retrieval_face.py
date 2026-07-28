from __future__ import annotations

from torch import nn

from ._common import build_baseline_face_retriever, smoke_test_face_retriever


_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_retrieval_face_tiny": {"width": 24, "depth": 1, "embedding_dim": 48},
    "mamba_retrieval_face_small": {"width": 36, "depth": 2, "embedding_dim": 64},
    "mamba_retrieval_face_base": {"width": 48, "depth": 3, "embedding_dim": 96},
}


def build_mamba_retrieval_face_face_retriever(
    *,
    in_channels: int,
    variant: str = "mamba_retrieval_face_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_face_retriever(
        family="mamba_retrieval_face",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_face_retriever(
        build_mamba_retrieval_face_face_retriever, "mamba_retrieval_face_tiny"
    )
