from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d

from ._rnn_common import AdditiveTokenAttention, parse_num_layers_suffix, pool_sequence


@dataclass(frozen=True)
class QRNNConfig:
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
    kernel_size: int


class QRNNLayer(nn.Module):
    """A simplified QRNN layer with fo-pooling."""

    def __init__(self, in_dim: int, hidden_dim: int, *, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        k = int(kernel_size)
        if k <= 0:
            raise ValueError("kernel_size must be > 0")
        self.kernel_size = k
        self.conv = nn.Conv1d(self.in_dim, 3 * self.hidden_dim, kernel_size=k, padding=k // 2)
        self.drop = nn.Dropout(p=float(dropout))

    def _pool(
        self, z: torch.Tensor, f: torch.Tensor, o: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # z,f,o: (B, T, H), mask: (B, T)
        b, t, h = z.shape
        c = torch.zeros((b, h), device=z.device, dtype=torch.float32)
        out = torch.zeros((b, t, h), device=z.device, dtype=torch.float32)
        for i in range(t):
            m = mask[:, i].view(b, 1).to(torch.float32)
            c_new = f[:, i] * c + (1.0 - f[:, i]) * z[:, i]
            c = m * c_new + (1.0 - m) * c
            h_i = o[:, i] * c
            out[:, i] = m * h_i
        return out

    def _scan(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        y = self.conv(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()  # (B,T',3H)
        # For even kernel sizes, symmetric padding yields T' = T + 1. Crop back to T to keep
        # token/mask alignment consistent across layers.
        t = int(mask.shape[1])
        y = y[:, :t, :].contiguous()
        z, f, o = y.chunk(3, dim=-1)
        z = torch.tanh(z)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        out = self._pool(z, f, o, mask)
        return self.drop(out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, reverse: bool) -> torch.Tensor:
        if reverse:
            y = self._scan(x.flip(1), mask.flip(1))
            return y.flip(1)
        return self._scan(x, mask)


class QRNNTextClassifier(nn.Module):
    def __init__(self, cfg: QRNNConfig) -> None:
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
            layers_fwd.append(
                QRNNLayer(
                    in_dim,
                    int(h),
                    kernel_size=int(cfg.kernel_size),
                    dropout=float(cfg.dropout),
                )
            )
            if self.bidirectional:
                layers_bwd.append(
                    QRNNLayer(
                        in_dim,
                        int(h),
                        kernel_size=int(cfg.kernel_size),
                        dropout=float(cfg.dropout),
                    )
                )
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


def build_qrnn_classifier(
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

    parts = rest.split("_")
    if len(parts) < 2 or parts[0] != "qrnn":
        raise ValueError("Expected variant like 'qrnn_last' or 'biqrnn_k3_attn2l'")

    kernel_size = 2
    i = 1
    if parts[1].startswith("k"):
        kernel_size = int(parts[1][1:])
        i = 2
    if i >= len(parts):
        raise ValueError("Missing pooling in QRNN variant")

    pooling = parts[i]
    pooling, num_layers = parse_num_layers_suffix(pooling)
    if pooling not in {"last", "mean", "max", "attn"}:
        raise ValueError("pooling must be one of: last|mean|max|attn")

    return QRNNTextClassifier(
        QRNNConfig(
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
            kernel_size=int(kernel_size),
        )
    )


def registry() -> dict[str, Builder]:
    pools = ("last", "mean", "max", "attn")

    r: dict[str, Builder] = {}

    # Family alias (historically `nl:qrnn`)
    r["qrnn"] = make_builder(build_qrnn_classifier, variant="qrnn_k2_mean")

    for prefix in ("qrnn", "biqrnn"):
        # k2/k3 support the full shallow+deep set (1..6 layers)
        for k in (2, 3):
            for pool in pools:
                for n_layers in (1, 2, 3, 4, 5, 6):
                    name = (
                        f"{prefix}_k{k}_{pool}"
                        if n_layers == 1
                        else f"{prefix}_k{k}_{pool}{n_layers}l"
                    )
                    r[name] = make_builder(build_qrnn_classifier, variant=name)

        # k4/k5 are only provided for deeper stacks (4..6 layers) to keep the zoo size bounded.
        for k in (4, 5):
            for pool in pools:
                for n_layers in (4, 5, 6):
                    name = f"{prefix}_k{k}_{pool}{n_layers}l"
                    r[name] = make_builder(build_qrnn_classifier, variant=name)

        # A couple extra wider kernels (explicit arch ids).
        for pool in ("mean", "attn"):
            name = f"{prefix}_k7_{pool}4l"
            r[name] = make_builder(build_qrnn_classifier, variant=name)

    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_qrnn_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="qrnn_k2_mean",
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


__all__ = ["QRNNTextClassifier", "QRNNConfig", "build_qrnn_classifier", "registry"]
