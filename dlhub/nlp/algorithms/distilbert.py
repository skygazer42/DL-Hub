
import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder

from ._transformer_core import TransformerConfig, TransformerTextClassifier


class DistilBertClassifier(TransformerTextClassifier):
    pass


def build_distilbert_classifier(
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
    if name not in {"distilbert_tiny", "distilbert"}:
        raise ValueError("Unknown DistilBERT variant. Supported: distilbert_tiny")

    embed_dim, heads, layers = 192, 4, 2

    return DistilBertClassifier(
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
            share_layers=False,
        )
    )

def registry() -> dict[str, Builder]:
    return {"distilbert_tiny": make_builder(build_distilbert_classifier, variant="distilbert_tiny")}

def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_distilbert_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="distilbert_tiny",
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


__all__ = ["DistilBertClassifier", "build_distilbert_classifier", "registry"]
