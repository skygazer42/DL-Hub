from __future__ import annotations

from torch import nn

from ._common import build_compact_vlm_family, smoke_test_vlm

_VARIANTS: dict[str, dict[str, int]] = {
    "vilt_tiny": {"width": 64, "depth": 2, "embed": 64},
    "vilt_small": {"width": 96, "depth": 3, "embed": 96},
    "vilt_base": {"width": 128, "depth": 4, "embed": 128},
}


def build_vilt_vlm(
    *,
    image_size: int = 32,
    vocab_size: int = 128,
    seq_len: int = 16,
    embed_dim: int = 64,
    num_classes: int = 0,
    variant: str = "vilt_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_compact_vlm_family(
        family="vilt",
        variants=_VARIANTS,
        image_size=int(image_size),
        vocab_size=int(vocab_size),
        seq_len=int(seq_len),
        embed_dim=int(embed_dim),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
        architecture_mode="single_stream",
    )


if __name__ == "__main__":
    smoke_test_vlm(build_vilt_vlm, "vilt_tiny")
