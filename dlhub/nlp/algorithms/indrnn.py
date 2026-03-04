from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d

from ._rnn_common import AdditiveTokenAttention, parse_num_layers_suffix, pool_sequence


@dataclass(frozen=True)
class IndRNNConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    hidden_dim: int
    bidirectional: bool
    pooling: str
    num_layers: int


class IndRNNLayer(nn.Module):
    """Independently Recurrent Neural Network (IndRNN), simplified."""

    def __init__(self, in_dim: int, hidden_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.in_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.u = nn.Parameter(torch.full((self.hidden_dim,), 0.5, dtype=torch.float32))
        self.drop = nn.Dropout(p=float(dropout))

    def _scan(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T)
        b, t, _ = x.shape
        pre = self.in_proj(x)  # (B,T,H)
        h = torch.zeros((b, self.hidden_dim), device=x.device, dtype=torch.float32)
        out = torch.zeros((b, t, self.hidden_dim), device=x.device, dtype=torch.float32)
        u = self.u.view(1, 1, self.hidden_dim).to(device=x.device)
        for i in range(t):
            m = mask[:, i].view(b, 1).to(torch.float32)
            h_new = torch.relu(pre[:, i] + u.squeeze(1) * h)
            h = m * h_new + (1.0 - m) * h
            out[:, i] = m * h
        return self.drop(out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, reverse: bool) -> torch.Tensor:
        if reverse:
            y = self._scan(x.flip(1), mask.flip(1))
            return y.flip(1)
        return self._scan(x, mask)


class IndRNNTextClassifier(nn.Module):
    def __init__(self, cfg: IndRNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        h = _d(int(cfg.hidden_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        self.embed = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.drop = nn.Dropout(p=float(cfg.dropout))

        self.bidirectional = bool(cfg.bidirectional)
        self.pooling = str(cfg.pooling).lower().strip()

        layers_fwd: list[nn.Module] = []
        layers_bwd: list[nn.Module] = []
        in_dim = int(d)
        for _ in range(int(cfg.num_layers)):
            layers_fwd.append(IndRNNLayer(in_dim, int(h), dropout=float(cfg.dropout)))
            if self.bidirectional:
                layers_bwd.append(IndRNNLayer(in_dim, int(h), dropout=float(cfg.dropout)))
                in_dim = 2 * int(h)
            else:
                in_dim = int(h)
        self.layers_fwd = nn.ModuleList(layers_fwd)
        self.layers_bwd = nn.ModuleList(layers_bwd)

        out_dim = int(h) * (2 if self.bidirectional else 1)
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

        x = self.drop(self.embed(input_ids).to(torch.float32))
        mask = attention_mask
        for i, layer in enumerate(self.layers_fwd):
            y_f = layer(x, mask, reverse=False)
            if self.bidirectional:
                y_b = self.layers_bwd[i](x, mask, reverse=True)
                x = torch.cat([y_f, y_b], dim=-1)
            else:
                x = y_f
            x = self.drop(x)

        pooled = pool_sequence(
            x,
            attention_mask,
            pooling=self.pooling,
            bidirectional=self.bidirectional,
            attn=self.attn,
        )
        return self.head(pooled)


def build_indrnn_classifier(
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
    bidirectional = False
    rest = name
    if rest.startswith("bi"):
        bidirectional = True
        rest = rest[2:]

    if not rest.startswith("indrnn_"):
        raise ValueError("Expected variant like 'indrnn_mean' or 'biindrnn_attn2l'")

    pooling = rest.split("_", 1)[1]
    pooling, num_layers = parse_num_layers_suffix(pooling)
    if pooling not in {"last", "mean", "max", "attn"}:
        raise ValueError("pooling must be one of: last|mean|max|attn")

    return IndRNNTextClassifier(
        IndRNNConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=96,
            hidden_dim=128,
            bidirectional=bool(bidirectional),
            pooling=str(pooling),
            num_layers=int(num_layers),
        )
    )


def registry() -> dict[str, Builder]:
    pools = ("last", "mean", "max", "attn")
    layers = (1, 2, 3, 4, 5, 6)

    r: dict[str, Builder] = {}

    # Family alias (historically `nl:indrnn`)
    r["indrnn"] = make_builder(build_indrnn_classifier, variant="indrnn_mean")

    for pool in pools:
        for n_layers in layers:
            name = f"indrnn_{pool}" if n_layers == 1 else f"indrnn_{pool}{n_layers}l"
            r[name] = make_builder(build_indrnn_classifier, variant=name)

            bi_name = f"biindrnn_{pool}" if n_layers == 1 else f"biindrnn_{pool}{n_layers}l"
            r[bi_name] = make_builder(build_indrnn_classifier, variant=bi_name)

    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_indrnn_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="indrnn_mean2l",
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


__all__ = ["IndRNNTextClassifier", "IndRNNConfig", "build_indrnn_classifier", "registry"]
