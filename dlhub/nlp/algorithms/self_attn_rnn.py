from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d, sequence_lengths


@dataclass(frozen=True)
class SelfAttnRNNConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    hidden_dim: int
    num_layers: int
    attn_dim: int
    num_hops: int


class SelfAttentiveRNNClassifier(nn.Module):
    """Structured self-attention on top of a BiLSTM encoder (Lin et al.), simplified."""

    def __init__(self, cfg: SelfAttnRNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        h = _d(int(cfg.hidden_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        a = _d(int(cfg.attn_dim), float(cfg.width_mult), min_dim=32, divisor=8)

        hops = int(cfg.num_hops)
        if hops <= 0:
            raise ValueError("num_hops must be > 0")
        self.num_hops = hops

        self.embedding = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.drop = nn.Dropout(p=float(cfg.dropout))

        self.encoder = nn.LSTM(
            input_size=int(d),
            hidden_size=int(h),
            num_layers=int(cfg.num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(cfg.dropout) if int(cfg.num_layers) > 1 else 0.0,
        )

        out_dim = int(h) * 2
        self.attn = nn.Sequential(
            nn.Linear(out_dim, int(a)),
            nn.Tanh(),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(int(a), int(hops), bias=False),
        )
        self.head = nn.Sequential(
            nn.Linear(int(hops) * out_dim, out_dim),
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
        x = self.drop(self.embedding(input_ids).to(torch.float32))
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.encoder(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=int(self.cfg.max_length)
        )
        out = self.drop(out)  # (B, T, 2H)

        scores = self.attn(out)  # (B, T, R)
        scores = scores.masked_fill(~attention_mask.to(torch.bool).unsqueeze(-1), -1e9)
        w = torch.softmax(scores, dim=1)  # (B, T, R)

        m = torch.matmul(w.transpose(1, 2), out)  # (B, R, 2H)
        feat = m.reshape(out.shape[0], -1)
        feat = self.drop(feat)
        return self.head(feat)


def build_self_attn_rnn_classifier(
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
    if name in {"self_attn_rnn_tiny", "self_attn_rnn"}:
        embed_dim, hidden_dim, layers, attn_dim, hops = 96, 128, 1, 128, 4
    elif name in {"self_attn_rnn_small"}:
        embed_dim, hidden_dim, layers, attn_dim, hops = 96, 160, 1, 160, 6
    elif name in {"self_attn_rnn_base"}:
        embed_dim, hidden_dim, layers, attn_dim, hops = 96, 192, 2, 192, 8
    else:
        raise ValueError(
            "Unknown SelfAttnRNN variant. Supported: self_attn_rnn_tiny|self_attn_rnn_small|self_attn_rnn_base"
        )

    return SelfAttentiveRNNClassifier(
        SelfAttnRNNConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(layers),
            attn_dim=int(attn_dim),
            num_hops=int(hops),
        )
    )


def registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}
    r["self_attn_rnn"] = make_builder(build_self_attn_rnn_classifier, variant="self_attn_rnn_tiny")
    for name in ("self_attn_rnn_tiny", "self_attn_rnn_small", "self_attn_rnn_base"):
        r[name] = make_builder(build_self_attn_rnn_classifier, variant=name)
    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_self_attn_rnn_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="self_attn_rnn_tiny",
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
    "SelfAttentiveRNNClassifier",
    "SelfAttnRNNConfig",
    "build_self_attn_rnn_classifier",
    "registry",
]

