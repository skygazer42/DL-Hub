
from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d


class AdditiveAttentionPool(nn.Module):
    def __init__(self, dim: int, *, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.proj = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(d, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C), mask: (B, T)
        scores = self.proj(x).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask.to(torch.bool), -1e9)
        w = torch.softmax(scores, dim=1)
        return (w.unsqueeze(-1) * x).sum(dim=1)


@dataclass(frozen=True)
class AttnPoolConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    num_heads: int


class AttentionPoolTextClassifier(nn.Module):
    def __init__(self, cfg: AttnPoolConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        self.embedding = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.drop = nn.Dropout(p=float(cfg.dropout))

        heads = int(cfg.num_heads)
        if heads <= 0:
            raise ValueError("num_heads must be > 0")
        if d % heads != 0:
            heads = 1
        self.num_heads = int(heads)
        self.head_dim = int(d // self.num_heads)

        self.q = nn.Parameter(torch.zeros(1, self.num_heads, 1, self.head_dim))
        self.kv = nn.Linear(int(d), 2 * int(d), bias=False)
        self.out = nn.Linear(int(d), int(d), bias=False)
        self.dropout = nn.Dropout(p=float(cfg.dropout))
        self.head = nn.Linear(int(d), int(cfg.num_classes))

        nn.init.normal_(self.q, std=0.02)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.cfg.pad_id)).to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)
        key_mask = attention_mask.to(torch.bool)

        x = self.embedding(input_ids).to(torch.float32)  # (B, T, D)
        x = self.drop(x)

        b, t, d = x.shape
        kv = self.kv(x)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q.expand(b, -1, -1, -1)  # (B, H, 1, Hd)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (float(self.head_dim) ** -0.5)  # (B,H,1,T)
        scores = scores.masked_fill(~key_mask.view(b, 1, 1, t), -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        pooled = torch.matmul(attn, v)  # (B,H,1,Hd)
        pooled = pooled.transpose(1, 2).contiguous().view(b, 1, d).squeeze(1)  # (B,D)
        pooled = self.out(pooled)
        pooled = self.drop(pooled)
        return self.head(pooled)


def build_attention_pool_classifier(
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
    if name in {"attn_pool", "attention_pool"}:
        embed_dim, heads = 64, 1
    elif name in {"mh_attn_pool", "multihead_attn_pool"}:
        embed_dim, heads = 128, 4
    else:
        raise ValueError("Unknown attention pool variant. Supported: attn_pool|mh_attn_pool")

    return AttentionPoolTextClassifier(
        AttnPoolConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            num_heads=int(heads),
        )
    )


def registry() -> dict[str, Builder]:
    return {
        "attn_pool": make_builder(build_attention_pool_classifier, variant="attn_pool"),
        "mh_attn_pool": make_builder(build_attention_pool_classifier, variant="mh_attn_pool"),
    }


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_attention_pool_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=1.0,
        dropout=0.1,
        variant="mh_attn_pool",
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
    "AdditiveAttentionPool",
    "AttentionPoolTextClassifier",
    "AttnPoolConfig",
    "build_attention_pool_classifier",
    "registry",
]
