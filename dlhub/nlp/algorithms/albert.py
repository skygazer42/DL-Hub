from __future__ import annotations

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder

from ._transformer_core import TransformerConfig, TransformerTextClassifier


class AlbertClassifier(TransformerTextClassifier):
    pass


_VARIANTS: dict[str, dict[str, int | bool]] = {
    "albert_tiny": {"embed_dim": 192, "num_heads": 4, "num_layers": 4, "share_layers": True},
}


def build_albert_classifier(
    *,
    vocab_size: int,
    pad_id: int,
    max_length: int,
    num_classes: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name == "albert":
        name = "albert_tiny"
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ALBERT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    return AlbertClassifier(
        TransformerConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(spec["embed_dim"]),
            num_heads=int(spec["num_heads"]),
            num_layers=int(spec["num_layers"]),
            pos="learned",
            rope=False,
            alibi=False,
            rel_bias=False,
            attn_impl="full",
            num_kv_heads=None,
            linformer_k=32,
            longformer_window=8,
            ffn_kind="gelu",
            norm_kind="layer",
            prenorm=False,  # post-norm encoder style
            causal=False,
            pool="cls",
            share_layers=bool(spec["share_layers"]),
        )
    )


def registry() -> dict[str, Builder]:
    return {
        "albert": make_builder(build_albert_classifier, variant="albert_tiny"),
        "albert_tiny": make_builder(build_albert_classifier, variant="albert_tiny"),
    }


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_albert_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="albert_tiny",
    )
    model.eval()

    x = torch.randint(0, vocab_size, (2, max_length), dtype=torch.long)
    attention_mask = torch.ones((2, max_length), dtype=torch.float32)
    with torch.no_grad():
        y = model({"input_ids": x, "attention_mask": attention_mask})

    n_params = sum(int(p.numel()) for p in model.parameters())
    print(f"smoke_ok: y.shape={tuple(y.shape)} params={n_params}")


if __name__ == "__main__":
    _smoke()


__all__ = ["AlbertClassifier", "build_albert_classifier", "registry"]
