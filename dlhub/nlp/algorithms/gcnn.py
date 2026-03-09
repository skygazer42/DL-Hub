
from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d


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
class GatedCNNConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    channels: int
    depth: int


class GatedCNNClassifier(nn.Module):
    """Gated CNN (GLU), simplified."""

    def __init__(self, cfg: GatedCNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        c = _d(int(cfg.channels), float(cfg.width_mult), min_dim=32, divisor=8)

        self.embedding = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.in_proj = nn.Conv1d(int(d), int(c), kernel_size=1)
        self.convs = nn.ModuleList(
            [nn.Conv1d(int(c), 2 * int(c), kernel_size=3, padding=1) for _ in range(int(cfg.depth))]
        )
        self.drop = nn.Dropout(p=float(cfg.dropout))
        self.head = nn.Linear(int(c), int(cfg.num_classes))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.cfg.pad_id)).to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)

        x = self.embedding(input_ids).to(torch.float32).transpose(1, 2).contiguous()  # (B, D, T)
        x = self.in_proj(x)
        for conv in self.convs:
            y = conv(x)
            a, b = y.chunk(2, dim=1)
            x = x + a * torch.sigmoid(b)
            x = self.drop(x)
        pooled = _masked_max_pool_1d(x, attention_mask)
        pooled = self.drop(pooled)
        return self.head(pooled)


def build_gated_cnn_classifier(
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
    if name in {"gcnn_tiny", "gated_cnn"}:
        channels, depth = 128, 4
    elif name in {"gcnn_small"}:
        channels, depth = 160, 6
    else:
        raise ValueError("Unknown GCNN variant. Supported: gcnn_tiny|gcnn_small")

    return GatedCNNClassifier(
        GatedCNNConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=96,
            channels=int(channels),
            depth=int(depth),
        )
    )


def registry() -> dict[str, Builder]:
    variants = ("gcnn_tiny", "gcnn_small")
    return {name: make_builder(build_gated_cnn_classifier, variant=name) for name in variants}


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_gated_cnn_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="gcnn_tiny",
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


__all__ = ["GatedCNNClassifier", "GatedCNNConfig", "build_gated_cnn_classifier", "registry"]
