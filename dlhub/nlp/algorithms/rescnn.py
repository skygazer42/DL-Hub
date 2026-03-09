
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


class _ConvBNReLU1d(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int, dropout: float) -> None:
        k = int(kernel_size)
        super().__init__(
            nn.Conv1d(int(in_ch), int(out_ch), kernel_size=k, padding=k // 2, bias=False),
            nn.BatchNorm1d(int(out_ch)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )


class ResCNNBlock(nn.Module):
    def __init__(self, channels: int, *, dropout: float) -> None:
        super().__init__()
        c = int(channels)
        self.net = nn.Sequential(
            _ConvBNReLU1d(c, c, kernel_size=3, dropout=float(dropout)),
            nn.Conv1d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(c),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


@dataclass(frozen=True)
class ResCNNConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    channels: int
    depth: int
    downsample_every: int


class ResCNNTextClassifier(nn.Module):
    """Residual 1D CNN classifier for token sequences."""

    def __init__(self, cfg: ResCNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        c = _d(int(cfg.channels), float(cfg.width_mult), min_dim=32, divisor=8)

        self.embedding = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.stem = _ConvBNReLU1d(int(d), int(c), kernel_size=3, dropout=float(cfg.dropout))

        blocks: list[nn.Module] = []
        for _ in range(int(cfg.depth)):
            blocks.append(ResCNNBlock(int(c), dropout=float(cfg.dropout)))
        self.blocks = nn.ModuleList(blocks)

        self.drop = nn.Dropout(p=float(cfg.dropout))
        self.head = nn.Sequential(
            nn.Linear(int(c), int(c)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(int(c), int(cfg.num_classes)),
        )

        self.downsample_every = max(1, int(cfg.downsample_every))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.cfg.pad_id)).to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)

        x = self.embedding(input_ids).to(torch.float32).transpose(1, 2).contiguous()  # (B, D, T)
        x = self.stem(x)  # (B, C, T)

        mask = attention_mask
        for i, blk in enumerate(self.blocks, start=1):
            x = blk(x)
            if i % self.downsample_every == 0 and x.shape[-1] > 1:
                x = torch.nn.functional.max_pool1d(x, kernel_size=3, stride=2, padding=1)
                mask = mask[:, ::2]

        pooled = _masked_max_pool_1d(x, mask)
        pooled = self.drop(pooled)
        return self.head(pooled)


def build_rescnn_classifier(
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
    if name in {"rescnn_tiny", "rescnn"}:
        channels, depth, down = 128, 4, 2
    elif name in {"rescnn_small"}:
        channels, depth, down = 160, 6, 2
    elif name in {"rescnn_base"}:
        channels, depth, down = 192, 8, 2
    else:
        raise ValueError("Unknown ResCNN variant. Supported: rescnn_tiny|rescnn_small|rescnn_base")

    return ResCNNTextClassifier(
        ResCNNConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=96,
            channels=int(channels),
            depth=int(depth),
            downsample_every=int(down),
        )
    )


def registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}
    r["rescnn"] = make_builder(build_rescnn_classifier, variant="rescnn_tiny")
    for name in ("rescnn_tiny", "rescnn_small", "rescnn_base"):
        r[name] = make_builder(build_rescnn_classifier, variant=name)
    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_rescnn_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="rescnn_tiny",
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


__all__ = ["ResCNNTextClassifier", "ResCNNConfig", "build_rescnn_classifier", "registry"]

