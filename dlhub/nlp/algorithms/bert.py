
import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder

from ._transformer_core import TransformerConfig, TransformerTextClassifier


class BertClassifier(TransformerTextClassifier):
    pass


_VARIANTS: dict[str, dict[str, int | str]] = {
    "bert_tiny": {"embed_dim": 192, "num_heads": 4, "num_layers": 2},
    "bert_small": {"embed_dim": 256, "num_heads": 4, "num_layers": 3},
    "bert_base": {"embed_dim": 320, "num_heads": 5, "num_layers": 4},
}


def build_bert_classifier(
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
    if name == "bert":
        name = "bert_tiny"
    if name not in _VARIANTS:
        raise ValueError(f"Unknown BERT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    return BertClassifier(
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
            share_layers=False,
        )
    )

def registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}

    # Family alias (historically `nl:bert`)
    r["bert"] = make_builder(build_bert_classifier, variant="bert_tiny")

    for name in _VARIANTS:
        r[name] = make_builder(build_bert_classifier, variant=name)

    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_bert_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="bert_tiny",
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


__all__ = ["BertClassifier", "build_bert_classifier", "registry"]
