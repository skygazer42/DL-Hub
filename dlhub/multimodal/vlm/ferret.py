from __future__ import annotations

from torch import nn

from ._common import build_toy_vlm_family, smoke_test_vlm

_VARIANTS: dict[str, dict[str, int]] = {
    "ferret_tiny": {"width": 64, "depth": 2, "embed": 64},
    "ferret_small": {"width": 96, "depth": 3, "embed": 96},
    "ferret_base": {"width": 128, "depth": 4, "embed": 128},
}


def build_ferret_vlm(
    *,
    image_size: int = 32,
    vocab_size: int = 128,
    seq_len: int = 16,
    embed_dim: int = 64,
    num_classes: int = 0,
    variant: str = "ferret_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    return build_toy_vlm_family(
        family="ferret",
        variants=_VARIANTS,
        image_size=int(image_size),
        vocab_size=int(vocab_size),
        seq_len=int(seq_len),
        embed_dim=int(embed_dim),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
        architecture_mode="bridge",
        use_instruction=True,
        use_query_bridge=True,
        use_generation_head=False,
    )


if __name__ == "__main__":
    smoke_test_vlm(build_ferret_vlm, "ferret_tiny")
