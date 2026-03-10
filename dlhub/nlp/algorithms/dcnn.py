import math
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


def _kmax_pool_1d(
    x: torch.Tensor, mask: torch.Tensor, *, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # x: (B, C, T), mask: (B, T) float
    if x.ndim != 3:
        raise ValueError(f"x must be (B, C, T), got {tuple(x.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be (B, T), got {tuple(mask.shape)}")
    b, c, t = x.shape
    kk = int(k)
    if kk <= 0:
        raise ValueError("k must be > 0")
    kk = min(kk, int(t))

    key_mask = mask.to(torch.bool).unsqueeze(1)  # (B, 1, T)
    x_masked = x.masked_fill(~key_mask, float("-inf"))
    vals, idx = x_masked.topk(kk, dim=-1)  # (B, C, k)

    # Order-preserving k-max pooling.
    idx_sorted = idx.sort(dim=-1).values
    pooled = x_masked.gather(-1, idx_sorted)

    # Build a conservative mask: if all channels are -inf at a pooled time index, mark invalid.
    valid = torch.isfinite(pooled).any(dim=1).to(torch.float32)  # (B, k)
    pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
    return pooled, valid


@dataclass(frozen=True)
class DCNNConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    channels: int
    depth: int
    k_top: int


class DCNNClassifier(nn.Module):
    """Dynamic CNN (Kalchbrenner et al.), simplified for fixed-length token inputs."""

    def __init__(self, cfg: DCNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        c = _d(int(cfg.channels), float(cfg.width_mult), min_dim=32, divisor=8)

        self.embedding = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.in_proj = nn.Conv1d(int(d), int(c), kernel_size=1)

        blocks: list[nn.Module] = []
        for _ in range(int(cfg.depth)):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(int(c), int(c), kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(int(c)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=float(cfg.dropout)),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.drop = nn.Dropout(p=float(cfg.dropout))
        self.head = nn.Sequential(
            nn.Linear(int(c), int(c)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(int(c), int(cfg.num_classes)),
        )

        self.k_top = int(cfg.k_top)
        if self.k_top <= 0:
            raise ValueError("k_top must be > 0")

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.cfg.pad_id)).to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)

        x = self.embedding(input_ids).to(torch.float32).transpose(1, 2).contiguous()  # (B, D, T)
        x = self.in_proj(x)  # (B, C, T)
        mask = attention_mask

        for blk in self.blocks:
            x = blk(x)
            # Dynamic k-max: halve sequence length each layer, but don't go below k_top.
            t = int(x.shape[-1])
            k = max(self.k_top, int(math.ceil(t / 2)))
            x, mask = _kmax_pool_1d(x, mask, k=k)

        pooled = _masked_max_pool_1d(x, mask)
        pooled = self.drop(pooled)
        return self.head(pooled)


def build_dcnn_classifier(
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
    if name in {"dcnn_tiny", "dcnn"}:
        channels, depth, k_top = 128, 4, 4
    elif name in {"dcnn_small"}:
        channels, depth, k_top = 160, 5, 4
    elif name in {"dcnn_base"}:
        channels, depth, k_top = 192, 6, 4
    else:
        raise ValueError("Unknown DCNN variant. Supported: dcnn_tiny|dcnn_small|dcnn_base")

    return DCNNClassifier(
        DCNNConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=96,
            channels=int(channels),
            depth=int(depth),
            k_top=int(k_top),
        )
    )


def registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}
    r["dcnn"] = make_builder(build_dcnn_classifier, variant="dcnn_tiny")
    for name in ("dcnn_tiny", "dcnn_small", "dcnn_base"):
        r[name] = make_builder(build_dcnn_classifier, variant=name)
    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_dcnn_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="dcnn_tiny",
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


__all__ = ["DCNNClassifier", "DCNNConfig", "build_dcnn_classifier", "registry"]
