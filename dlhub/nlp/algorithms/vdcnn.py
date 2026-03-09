
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


class _ConvBlock1d(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int, dropout: float) -> None:
        super().__init__(
            nn.Conv1d(
                int(in_ch),
                int(out_ch),
                kernel_size=int(kernel_size),
                padding=int(kernel_size) // 2,
                bias=False,
            ),
            nn.BatchNorm1d(int(out_ch)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )


@dataclass(frozen=True)
class VDCNNConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    channels: int
    num_blocks: int


class VDCNNClassifier(nn.Module):
    """Very Deep CNN (VDCNN), simplified (token-embedding conv stack)."""

    def __init__(self, cfg: VDCNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        c = _d(int(cfg.channels), float(cfg.width_mult), min_dim=32, divisor=8)

        self.embedding = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.stem = _ConvBlock1d(int(d), int(c), kernel_size=3, dropout=float(cfg.dropout))

        blocks: list[nn.Module] = []
        for _ in range(int(cfg.num_blocks)):
            blocks.append(_ConvBlock1d(int(c), int(c), kernel_size=3, dropout=float(cfg.dropout)))
            blocks.append(_ConvBlock1d(int(c), int(c), kernel_size=3, dropout=float(cfg.dropout)))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(int(c), int(c)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(int(c), int(cfg.num_classes)),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.cfg.pad_id)).to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)

        x = self.embedding(input_ids).to(torch.float32).transpose(1, 2).contiguous()
        x = self.stem(x)
        x = self.blocks(x)
        pooled = _masked_max_pool_1d(x, attention_mask)
        return self.head(pooled)


def build_vdcnn_classifier(
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
    if name in {"vdcnn_tiny", "vdcnn"}:
        channels, num_blocks = 128, 4
    elif name in {"vdcnn_small"}:
        channels, num_blocks = 160, 6
    elif name in {"vdcnn_base"}:
        channels, num_blocks = 192, 8
    else:
        raise ValueError("Unknown VDCNN variant. Supported: vdcnn_tiny|vdcnn_small|vdcnn_base")

    return VDCNNClassifier(
        VDCNNConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=96,
            channels=int(channels),
            num_blocks=int(num_blocks),
        )
    )


def registry() -> dict[str, Builder]:
    variants = ("vdcnn_tiny", "vdcnn_small", "vdcnn_base")
    return {name: make_builder(build_vdcnn_classifier, variant=name) for name in variants}


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_vdcnn_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="vdcnn_tiny",
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


__all__ = ["VDCNNClassifier", "VDCNNConfig", "build_vdcnn_classifier", "registry"]
