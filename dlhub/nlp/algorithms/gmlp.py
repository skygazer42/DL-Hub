from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d, masked_mean_pool


class SpatialGatingUnit(nn.Module):
    def __init__(self, num_tokens: int, dim: int, *, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.proj = nn.Linear(int(num_tokens), int(num_tokens))
        self.drop = nn.Dropout(p=float(dropout))
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(1, 2)).transpose(1, 2)
        v = self.drop(v)
        return u * v


class GMLPBlock(nn.Module):
    def __init__(self, dim: int, *, num_tokens: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        hidden = max(8, d * 2)
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 2 * hidden)
        self.act = nn.GELU()
        self.sgu = SpatialGatingUnit(num_tokens=int(num_tokens), dim=hidden, dropout=float(dropout))
        self.fc2 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.sgu(y)
        y = self.fc2(y)
        return x + self.drop(y)


@dataclass(frozen=True)
class GMLPConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int


class GMLPTextClassifier(nn.Module):
    def __init__(self, cfg: GMLPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=64, divisor=8)
        self.token = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.pos = nn.Embedding(int(cfg.max_length), int(d))
        self.drop = nn.Dropout(p=float(cfg.dropout))
        self.blocks = nn.Sequential(
            *[
                GMLPBlock(int(d), num_tokens=int(cfg.max_length), dropout=float(cfg.dropout))
                for _ in range(int(cfg.depth))
            ]
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


def build_gmlp_classifier(
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
    if name in {"gmlp_tiny", "gmlp"}:
        embed_dim, depth = 192, 2
    elif name in {"gmlp_small"}:
        embed_dim, depth = 256, 3
    elif name in {"gmlp_base"}:
        embed_dim, depth = 320, 4
    else:
        raise ValueError("Unknown gMLP variant. Supported: gmlp_tiny|gmlp_small|gmlp_base")

    return GMLPTextClassifier(
        GMLPConfig(
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
    r["gmlp"] = make_builder(build_gmlp_classifier, variant="gmlp_tiny")
    for name in ("gmlp_tiny", "gmlp_small", "gmlp_base"):
        r[name] = make_builder(build_gmlp_classifier, variant=name)
    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_gmlp_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="gmlp_tiny",
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


__all__ = ["GMLPTextClassifier", "GMLPConfig", "build_gmlp_classifier", "registry"]
