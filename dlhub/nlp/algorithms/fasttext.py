from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d, masked_mean_pool


@dataclass(frozen=True)
class FastTextNGramConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    ngram_buckets: int
    use_bigrams: bool
    use_trigrams: bool


class FastTextNGramClassifier(nn.Module):
    """A fastText-style n-gram classifier (hash n-grams from token ids)."""

    def __init__(self, cfg: FastTextNGramConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        self.d = int(d)
        buckets = int(cfg.ngram_buckets)
        if buckets <= 0:
            raise ValueError("ngram_buckets must be > 0")
        self.buckets = buckets

        self.embedding = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.ngram_embed = nn.Embedding(int(cfg.ngram_buckets), int(d))
        self.drop = nn.Dropout(p=float(cfg.dropout))
        self.head = nn.Linear(int(d), int(cfg.num_classes))

        self.use_bigrams = bool(cfg.use_bigrams)
        self.use_trigrams = bool(cfg.use_trigrams)
        if not (self.use_bigrams or self.use_trigrams):
            raise ValueError("At least one of use_bigrams/use_trigrams must be True")

    def _hash_bigram(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * 1315423911 + b * 2654435761) % int(self.buckets)

    def _hash_trigram(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return (a * 1315423911 + b * 2654435761 + c * 97531) % int(self.buckets)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)  # (B, T)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.cfg.pad_id)).to(torch.float32)
        mask = attention_mask.to(torch.float32)

        emb = self.embedding(input_ids)  # (B, T, D)
        feats = [emb]

        ids = input_ids.clamp(min=0)
        if self.use_bigrams and ids.shape[1] >= 2:
            a = ids[:, :-1]
            b = ids[:, 1:]
            h = self._hash_bigram(a, b)
            feats.append(self.ngram_embed(h))
            mask = mask[:, :-1]
        if self.use_trigrams and ids.shape[1] >= 3:
            a = ids[:, :-2]
            b = ids[:, 1:-1]
            c = ids[:, 2:]
            h = self._hash_trigram(a, b, c)
            feats.append(self.ngram_embed(h))
            if mask.shape[1] != h.shape[1]:
                mask = mask[:, : h.shape[1]]

        x = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
        if x.shape[1] != mask.shape[1]:
            # Align mask to concatenated features length (best-effort).
            if x.shape[1] > mask.shape[1]:
                pad = x.shape[1] - mask.shape[1]
                mask = torch.nn.functional.pad(mask, (0, pad), value=0.0)
            else:
                mask = mask[:, : x.shape[1]]

        pooled = masked_mean_pool(x, mask)
        pooled = self.drop(pooled)
        return self.head(pooled)


def build_fasttext_ngram_classifier(
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
    if name in {"fasttext_bigram", "fasttext_bi"}:
        use_bi, use_tri = True, False
    elif name in {"fasttext_trigram", "fasttext_tri"}:
        use_bi, use_tri = False, True
    elif name in {"fasttext_bi_tri", "fasttext"}:
        use_bi, use_tri = True, True
    else:
        raise ValueError(
            "Unknown fastText variant. Supported: fasttext_bigram|fasttext_trigram|fasttext_bi_tri"
        )

    return FastTextNGramClassifier(
        FastTextNGramConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=64,
            ngram_buckets=2048,
            use_bigrams=bool(use_bi),
            use_trigrams=bool(use_tri),
        )
    )


def registry() -> dict[str, Builder]:
    return {
        "fasttext_bigram": make_builder(build_fasttext_ngram_classifier, variant="fasttext_bigram"),
        "fasttext_trigram": make_builder(build_fasttext_ngram_classifier, variant="fasttext_trigram"),
        "fasttext_bi_tri": make_builder(build_fasttext_ngram_classifier, variant="fasttext_bi_tri"),
    }


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_fasttext_ngram_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=1.0,
        dropout=0.1,
        variant="fasttext_bi_tri",
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


__all__ = [
    "FastTextNGramClassifier",
    "FastTextNGramConfig",
    "build_fasttext_ngram_classifier",
    "registry",
]
