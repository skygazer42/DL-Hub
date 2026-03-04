from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d, masked_mean_pool


class FeedForward(nn.Module):
    def __init__(self, dim: int, *, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        hidden = max(8, int(round(d * float(mlp_ratio))))
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden, d),
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FNetBlock(nn.Module):
    def __init__(self, dim: int, *, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.drop1 = nn.Dropout(p=float(dropout))
        self.norm2 = nn.LayerNorm(d)
        self.ff = FeedForward(d, mlp_ratio=4.0, dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing via FFT (real part).
        y = self.norm1(x)
        y = torch.fft.rfft(y, dim=1)
        y = torch.fft.irfft(y, n=x.shape[1], dim=1)
        x = x + self.drop1(y)

        z = self.norm2(x)
        x = x + self.ff(z)
        return x


@dataclass(frozen=True)
class FNetConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int


class FNetTextClassifier(nn.Module):
    def __init__(self, cfg: FNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=64, divisor=8)
        self.token = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.pos = nn.Embedding(int(cfg.max_length), int(d))
        self.drop = nn.Dropout(p=float(cfg.dropout))
        self.blocks = nn.Sequential(
            *[FNetBlock(int(d), dropout=float(cfg.dropout)) for _ in range(int(cfg.depth))]
        )
        self.norm = nn.LayerNorm(int(d))
        self.head = nn.Linear(int(d), int(cfg.num_classes))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.cfg.pad_id)).to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)

        b, t = input_ids.shape
        if t != int(self.cfg.max_length):
            raise ValueError(f"Expected max_length={int(self.cfg.max_length)}, got T={t}")
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.token(input_ids) + self.pos(pos)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        pooled = masked_mean_pool(x, attention_mask)
        return self.head(pooled)


def build_fnet_classifier(
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
    if name in {"fnet_tiny", "fnet"}:
        embed_dim, depth = 192, 2
    elif name in {"fnet_small"}:
        embed_dim, depth = 256, 3
    elif name in {"fnet_base"}:
        embed_dim, depth = 320, 4
    else:
        raise ValueError("Unknown FNet variant. Supported: fnet_tiny|fnet_small|fnet_base")

    return FNetTextClassifier(
        FNetConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            depth=int(depth),
        )
    )


def registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}
    r["fnet"] = make_builder(build_fnet_classifier, variant="fnet_tiny")
    for name in ("fnet_tiny", "fnet_small", "fnet_base"):
        r[name] = make_builder(build_fnet_classifier, variant=name)
    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_fnet_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="fnet_tiny",
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


__all__ = ["FNetTextClassifier", "FNetConfig", "build_fnet_classifier", "registry"]
