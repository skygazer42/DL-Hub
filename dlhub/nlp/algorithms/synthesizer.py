
import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder

from ._transformer_core import TransformerConfig, TransformerTextClassifier


class SynthesizerClassifier(TransformerTextClassifier):
    pass


def build_synthesizer_classifier(
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

    if "mlp" in name:
        mode = "mlp"
    elif "random" in name or "rand" in name:
        mode = "random"
    else:
        # Default to the paper's random/dense synthesizer spirit for a stable baseline.
        mode = "random"

    if name.endswith("tiny") or name in {"synthesizer_tiny", "synthesizer"}:
        embed_dim, heads, layers = 192, 4, 2
        hidden = 128
    elif name.endswith("small"):
        embed_dim, heads, layers = 256, 4, 3
        hidden = 160
    elif name.endswith("base"):
        embed_dim, heads, layers = 320, 5, 4
        hidden = 192
    else:
        raise ValueError(
            "Unknown Synthesizer variant. Supported: synthesizer_random_tiny|synthesizer_mlp_small|synthesizer_random_base|..."
        )

    return SynthesizerClassifier(
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
            attn_impl="synthesizer",
            num_kv_heads=None,
            linformer_k=32,
            longformer_window=8,
            ffn_kind="gelu",
            norm_kind="layer",
            prenorm=True,
            causal=False,
            pool="mean",
            share_layers=False,
            synthesizer_mode=str(mode),
            synthesizer_hidden=int(hidden),
        )
    )

def registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}

    # Family alias (historically `nl:synthesizer`)
    r["synthesizer"] = make_builder(build_synthesizer_classifier, variant="synthesizer_random_tiny")

    for name in (
        "synthesizer_random_tiny",
        "synthesizer_random_small",
        "synthesizer_random_base",
        "synthesizer_mlp_tiny",
        "synthesizer_mlp_small",
        "synthesizer_mlp_base",
    ):
        r[name] = make_builder(build_synthesizer_classifier, variant=name)

    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_synthesizer_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="synthesizer_random_tiny",
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


__all__ = ["SynthesizerClassifier", "build_synthesizer_classifier", "registry"]
