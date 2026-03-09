
from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d, masked_max_pool, masked_mean_pool


@dataclass(frozen=True)
class PoolingConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    pool: str  # mean|max|meanmax


class PoolingTextClassifier(nn.Module):
    def __init__(self, cfg: PoolingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        self.embed_dim = int(d)
        self.pool = str(cfg.pool).lower().strip()
        if self.pool not in {"mean", "max", "meanmax"}:
            raise ValueError("pool must be one of: mean|max|meanmax")

        self.embedding = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.drop = nn.Dropout(p=float(cfg.dropout))

        head_in = int(d) if self.pool in {"mean", "max"} else int(2 * d)
        self.head = nn.Linear(head_in, int(cfg.num_classes))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)  # (B, T)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.cfg.pad_id)).to(torch.float32)
        else:
            attention_mask = attention_mask.to(torch.float32)

        x = self.embedding(input_ids)  # (B, T, D)
        if self.pool == "mean":
            pooled = masked_mean_pool(x, attention_mask)
        elif self.pool == "max":
            pooled = masked_max_pool(x, attention_mask)
        else:
            pooled = torch.cat(
                [masked_mean_pool(x, attention_mask), masked_max_pool(x, attention_mask)], dim=-1
            )
        pooled = self.drop(pooled)
        return self.head(pooled)


def build_pooling_classifier(
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
    if name in {"mean_pool", "mean"}:
        pool = "mean"
    elif name in {"max_pool", "max"}:
        pool = "max"
    elif name in {"meanmax_pool", "meanmax"}:
        pool = "meanmax"
    else:
        raise ValueError("Unknown pooling variant. Supported: mean_pool|max_pool|meanmax_pool")

    return PoolingTextClassifier(
        PoolingConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=64,
            pool=str(pool),
        )
    )


def registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}

    # Keep backward-compatible arch ids from the old one-file-per-variant layout.
    r["mean"] = make_builder(build_pooling_classifier, variant="mean_pool")
    r["mean_pool"] = make_builder(build_pooling_classifier, variant="mean_pool")

    r["max"] = make_builder(build_pooling_classifier, variant="max_pool")
    r["max_pool"] = make_builder(build_pooling_classifier, variant="max_pool")

    r["meanmax_pool"] = make_builder(build_pooling_classifier, variant="meanmax_pool")

    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_pooling_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=1.0,
        dropout=0.1,
        variant="mean_pool",
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


__all__ = ["PoolingTextClassifier", "PoolingConfig", "build_pooling_classifier", "registry"]
