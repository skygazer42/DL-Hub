from __future__ import annotations

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder

from ._transformer_core import TransformerConfig, TransformerTextClassifier


class PerformerClassifier(TransformerTextClassifier):
    pass


def build_performer_classifier(
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
    if name in {"performer_tiny", "performer"}:
        embed_dim, heads, layers = 192, 4, 2
    elif name in {"performer_small"}:
        embed_dim, heads, layers = 256, 4, 3
    elif name in {"performer_base"}:
        embed_dim, heads, layers = 320, 5, 4
    else:
        raise ValueError(
            "Unknown Performer variant. Supported: performer_tiny|performer_small|performer_base"
        )

    return PerformerClassifier(
        TransformerConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            num_heads=int(heads),
            num_layers=int(layers),
            pos="sin",
            rope=False,
            alibi=False,
            rel_bias=False,
            attn_impl="performer",
            num_kv_heads=None,
            linformer_k=32,
            longformer_window=8,
            ffn_kind="gelu",
            norm_kind="layer",
            prenorm=True,
            causal=False,
            pool="mean",
            share_layers=False,
        )
    )

def registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}
    r["performer"] = make_builder(build_performer_classifier, variant="performer_tiny")
    for name in ("performer_tiny", "performer_small", "performer_base"):
        r[name] = make_builder(build_performer_classifier, variant=name)
    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_performer_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="performer_tiny",
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


__all__ = ["PerformerClassifier", "build_performer_classifier", "registry"]
