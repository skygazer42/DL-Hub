from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d

_VARIANTS: tuple[str, ...] = (
    "tcn_base",
    "tcn_c128_d2",
    "tcn_c128_d3",
    "tcn_c128_d4",
    "tcn_c128_d5",
    "tcn_c160_d2",
    "tcn_c160_d3",
    "tcn_c160_d4",
    "tcn_c160_d5",
    "tcn_c192_d2",
    "tcn_c192_d3",
    "tcn_c192_d4",
    "tcn_c192_d5",
    "tcn_c224_d4",
    "tcn_c224_d5",
    "tcn_c256_d6",
    "tcn_c64_d2",
    "tcn_c64_d3",
    "tcn_c96_d2",
    "tcn_c96_d3",
    "tcn_c96_d4",
    "tcn_c96_d5",
    "tcn_small",
    "tcn_tiny",
)


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


class TCNBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int, dropout: float) -> None:
        super().__init__()
        c = int(channels)
        d = int(dilation)
        self.net = nn.Sequential(
            nn.Conv1d(c, c, kernel_size=3, padding=d, dilation=d, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Conv1d(c, c, kernel_size=3, padding=d, dilation=d, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


@dataclass(frozen=True)
class TCNConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    channels: int
    depth: int


class TCNClassifier(nn.Module):
    def __init__(self, cfg: TCNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        c = _d(int(cfg.channels), float(cfg.width_mult), min_dim=32, divisor=8)

        self.embedding = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.stem = _ConvBlock1d(int(d), int(c), kernel_size=3, dropout=float(cfg.dropout))
        self.blocks = nn.Sequential(
            *[TCNBlock(int(c), dilation=2**i, dropout=float(cfg.dropout)) for i in range(int(cfg.depth))]
        )
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


def build_tcn_classifier(
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
    if name in {"tcn_tiny", "tcn"}:
        channels, depth = 128, 3
    elif name in {"tcn_small"}:
        channels, depth = 160, 4
    elif name in {"tcn_base"}:
        channels, depth = 192, 5
    else:
        # Lab format: tcn_c128_d4 (channels + depth).
        parts = name.split("_")
        if (
            len(parts) == 3
            and parts[0] == "tcn"
            and parts[1].startswith("c")
            and parts[2].startswith("d")
            and parts[1][1:].isdigit()
            and parts[2][1:].isdigit()
        ):
            channels = int(parts[1][1:])
            depth = int(parts[2][1:])
            if channels <= 0 or depth <= 0:
                raise ValueError("Invalid TCN lab variant; channels and depth must be > 0")
        else:
            raise ValueError(
                "Unknown TCN variant. Supported: tcn_tiny|tcn_small|tcn_base|tcn_c<channels>_d<depth>"
            )

    return TCNClassifier(
        TCNConfig(
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
    return {name: make_builder(build_tcn_classifier, variant=name) for name in _VARIANTS}


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_tcn_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="tcn_tiny",
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


__all__ = ["TCNClassifier", "TCNConfig", "build_tcn_classifier", "registry"]
