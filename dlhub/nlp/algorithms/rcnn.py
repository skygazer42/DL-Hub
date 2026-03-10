from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d, sequence_lengths


def _masked_max_pool_1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # x: (B, C, T), mask: (B, T)
    if x.ndim != 3:
        raise ValueError(f"x must be (B, C, T), got {tuple(x.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be (B, T), got {tuple(mask.shape)}")
    key_mask = mask.to(torch.bool).unsqueeze(1)  # (B, 1, T)
    x = x.masked_fill(~key_mask, float("-inf"))
    pooled = x.max(dim=-1).values
    pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
    return pooled


@dataclass(frozen=True)
class RCNNConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    hidden_dim: int
    cell: str  # gru|lstm
    bidirectional: bool


class RCNNClassifier(nn.Module):
    """Recurrent CNN (Lai et al.), simplified for fixed-length token inputs."""

    def __init__(self, cfg: RCNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        h = _d(int(cfg.hidden_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        self.embed = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.drop = nn.Dropout(p=float(cfg.dropout))

        cell = str(cfg.cell).lower().strip()
        if cell == "gru":
            rnn_cls = nn.GRU
        elif cell == "lstm":
            rnn_cls = nn.LSTM
        else:
            raise ValueError("cell must be gru|lstm")
        self.cell = cell
        self.bidirectional = bool(cfg.bidirectional)

        self.rnn = rnn_cls(
            input_size=int(d),
            hidden_size=int(h),
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        out_dim = int(h) * (2 if self.bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(int(d) + out_dim, out_dim),
            nn.Tanh(),
            nn.Dropout(p=float(cfg.dropout)),
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
        emb = self.drop(self.embed(input_ids).to(torch.float32))
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.rnn(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=int(self.cfg.max_length)
        )
        out = self.drop(out)

        x = torch.cat([emb, out], dim=-1)
        x = self.proj(x)

        pooled = _masked_max_pool_1d(x.transpose(1, 2).contiguous(), attention_mask)
        return self.head(pooled)


def build_rcnn_classifier(
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
    if name in {"rcnn_gru", "rcnn"}:
        cell, bidirectional = "gru", True
    elif name in {"rcnn_lstm"}:
        cell, bidirectional = "lstm", True
    elif name in {"rcnn_gru_uni"}:
        cell, bidirectional = "gru", False
    elif name in {"rcnn_lstm_uni"}:
        cell, bidirectional = "lstm", False
    else:
        raise ValueError(
            "Unknown RCNN variant. Supported: rcnn_gru|rcnn_lstm|rcnn_gru_uni|rcnn_lstm_uni"
        )

    return RCNNClassifier(
        RCNNConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=96,
            hidden_dim=128,
            cell=str(cell),
            bidirectional=bool(bidirectional),
        )
    )


def registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}

    # Family alias (historically `nl:rcnn`)
    r["rcnn"] = make_builder(build_rcnn_classifier, variant="rcnn_gru")

    for name in ("rcnn_gru", "rcnn_lstm", "rcnn_gru_uni", "rcnn_lstm_uni"):
        r[name] = make_builder(build_rcnn_classifier, variant=name)

    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_rcnn_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="rcnn_gru",
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


__all__ = ["RCNNClassifier", "RCNNConfig", "build_rcnn_classifier", "registry"]
