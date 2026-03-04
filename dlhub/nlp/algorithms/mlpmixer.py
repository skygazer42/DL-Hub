from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d, masked_mean_pool


class MLPMixerBlock(nn.Module):
    def __init__(self, dim: int, *, num_tokens: int, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        t = int(num_tokens)
        self.norm1 = nn.LayerNorm(d)
        self.token_mlp = nn.Sequential(
            nn.Linear(t, t),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(t, t),
        )
        self.drop1 = nn.Dropout(p=float(dropout))

        self.norm2 = nn.LayerNorm(d)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(d * 4, d),
        )
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)  # (B, D, T)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + self.drop1(y)

        z = self.norm2(x)
        x = x + self.drop2(self.channel_mlp(z))
        return x


@dataclass(frozen=True)
class MLPMixerConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    depth: int


class MLPMixerTextClassifier(nn.Module):
    def __init__(self, cfg: MLPMixerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=64, divisor=8)
        self.token = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.pos = nn.Embedding(int(cfg.max_length), int(d))
        self.drop = nn.Dropout(p=float(cfg.dropout))
        self.blocks = nn.Sequential(
            *[
                MLPMixerBlock(int(d), num_tokens=int(cfg.max_length), dropout=float(cfg.dropout))
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


def build_mlp_mixer_classifier(
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
    if name in {"mlp_mixer_tiny", "mlpmixer_tiny", "mlp_mixer"}:
        embed_dim, depth = 192, 2
    elif name in {"mlp_mixer_small", "mlpmixer_small"}:
        embed_dim, depth = 256, 3
    elif name in {"mlp_mixer_base", "mlpmixer_base"}:
        embed_dim, depth = 320, 4
    else:
        raise ValueError(
            "Unknown MLP-Mixer variant. Supported: mlp_mixer_tiny|mlp_mixer_small|mlp_mixer_base"
        )

    return MLPMixerTextClassifier(
        MLPMixerConfig(
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

    # Family alias (historically `nl:mlpmixer`)
    r["mlpmixer"] = make_builder(build_mlp_mixer_classifier, variant="mlp_mixer_tiny")

    for name in ("mlp_mixer_tiny", "mlp_mixer_small", "mlp_mixer_base"):
        r[name] = make_builder(build_mlp_mixer_classifier, variant=name)

    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_mlp_mixer_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="mlp_mixer_tiny",
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


__all__ = ["MLPMixerTextClassifier", "MLPMixerConfig", "build_mlp_mixer_classifier", "registry"]
