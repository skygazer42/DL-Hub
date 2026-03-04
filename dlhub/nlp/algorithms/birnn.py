from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d, sequence_lengths

from ._rnn_common import AdditiveTokenAttention, parse_num_layers_suffix, pool_sequence


@dataclass(frozen=True)
class BiRNNConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    hidden_dim: int
    pooling: str  # last|max|mean|attn
    num_layers: int


class BiRNNTextClassifier(nn.Module):
    def __init__(self, cfg: BiRNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        h = _d(int(cfg.hidden_dim), float(cfg.width_mult), min_dim=32, divisor=8)

        self.embedding = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.drop = nn.Dropout(p=float(cfg.dropout))

        self.encoder = nn.RNN(
            input_size=int(d),
            hidden_size=int(h),
            num_layers=int(cfg.num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(cfg.dropout) if int(cfg.num_layers) > 1 else 0.0,
        )

        self.pooling = str(cfg.pooling).lower().strip()

        out_dim = int(h) * 2
        self.attn = (
            AdditiveTokenAttention(out_dim, dropout=float(cfg.dropout))
            if self.pooling == "attn"
            else None
        )

        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(out_dim, int(cfg.num_classes)),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.cfg.pad_id)).to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)

        lengths = sequence_lengths(attention_mask).cpu()
        emb = self.drop(self.embedding(input_ids).to(torch.float32))
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.encoder(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=int(self.cfg.max_length)
        )
        out = self.drop(out)

        pooled = pool_sequence(
            out,
            attention_mask,
            pooling=self.pooling,
            bidirectional=True,
            attn=self.attn,
        )
        return self.head(pooled)


def build_birnn_classifier(
    *,
    vocab_size: int,
    pad_id: int,
    max_length: int,
    num_classes: int,
    width_mult: float = 1.0,
    dropout: float = 0.2,
    variant: str,
) -> nn.Module:
    name = str(variant).lower().strip()
    if not name.startswith("birnn_"):
        raise ValueError("Expected variant like 'birnn_mean' or 'birnn_attn3l'")

    pooling = name.split("_", 1)[1]
    pooling, num_layers = parse_num_layers_suffix(pooling)
    if pooling not in {"last", "mean", "max", "attn"}:
        raise ValueError("pooling must be one of: last|mean|max|attn")

    return BiRNNTextClassifier(
        BiRNNConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=96,
            hidden_dim=128,
            pooling=str(pooling),
            num_layers=int(num_layers),
        )
    )


def registry() -> dict[str, Builder]:
    pools = ("last", "mean", "max", "attn")
    layers = (1, 2, 3, 4, 5, 6)

    r: dict[str, Builder] = {}
    for pool in pools:
        for n_layers in layers:
            name = f"birnn_{pool}" if n_layers == 1 else f"birnn_{pool}{n_layers}l"
            r[name] = make_builder(build_birnn_classifier, variant=name)
    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_birnn_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="birnn_mean2l",
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


__all__ = ["BiRNNTextClassifier", "BiRNNConfig", "build_birnn_classifier", "registry"]
