
from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d, sequence_lengths

from ._rnn_common import AdditiveTokenAttention, parse_num_layers_suffix, pool_sequence


@dataclass(frozen=True)
class CNNBiLSTMConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    conv_channels: int
    hidden_dim: int
    pooling: str  # last|mean|max|attn
    num_layers: int


class CNNBiLSTMTextClassifier(nn.Module):
    """CNN front-end + BiLSTM encoder classifier."""

    def __init__(self, cfg: CNNBiLSTMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        c = _d(int(cfg.conv_channels), float(cfg.width_mult), min_dim=32, divisor=8)
        h = _d(int(cfg.hidden_dim), float(cfg.width_mult), min_dim=32, divisor=8)

        self.embedding = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.cnn = nn.Sequential(
            nn.Conv1d(int(d), int(c), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(int(c)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
        )
        self.drop = nn.Dropout(p=float(cfg.dropout))

        self.encoder = nn.LSTM(
            input_size=int(c),
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
        emb = self.embedding(input_ids).to(torch.float32)  # (B, T, D)
        x = self.cnn(emb.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()  # (B, T, C)
        x = self.drop(x)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
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


def build_cnn_bilstm_classifier(
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
    if not name.startswith("cnn_bilstm_"):
        raise ValueError("Expected variant like 'cnn_bilstm_mean' or 'cnn_bilstm_attn3l'")

    pooling = name.split("_", 2)[2]
    pooling, num_layers = parse_num_layers_suffix(pooling)
    if pooling not in {"last", "mean", "max", "attn"}:
        raise ValueError("pooling must be one of: last|mean|max|attn")

    return CNNBiLSTMTextClassifier(
        CNNBiLSTMConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=96,
            conv_channels=128,
            hidden_dim=128,
            pooling=str(pooling),
            num_layers=int(num_layers),
        )
    )


def registry() -> dict[str, Builder]:
    pools = ("last", "mean", "max", "attn")
    layers = (1, 2, 3)

    r: dict[str, Builder] = {}

    # Family alias
    r["cnn_bilstm"] = make_builder(build_cnn_bilstm_classifier, variant="cnn_bilstm_mean")

    for pool in pools:
        for n_layers in layers:
            name = f"cnn_bilstm_{pool}" if n_layers == 1 else f"cnn_bilstm_{pool}{n_layers}l"
            r[name] = make_builder(build_cnn_bilstm_classifier, variant=name)

    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_cnn_bilstm_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="cnn_bilstm_attn2l",
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


__all__ = ["CNNBiLSTMTextClassifier", "CNNBiLSTMConfig", "build_cnn_bilstm_classifier", "registry"]

