from __future__ import annotations
import torch
from torch import nn


def check_btnf(x):
    x = x.to(torch.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected input shape (B,T,F), got {tuple(x.shape)}")
    return x


class CompactHandwritingRecognizer(nn.Module):
    def __init__(self, *, family: str, input_dim: int, vocab_size: int, width: int, depth: int):
        super().__init__()
        self.family = str(family)
        self.proj = nn.Linear(int(input_dim), int(width))
        self.rnn = nn.GRU(int(width), int(width), num_layers=max(1, int(depth)), batch_first=True)
        self.head = nn.Linear(int(width), int(vocab_size))

    def forward(self, strokes):
        x = check_btnf(strokes)
        seq, _ = self.rnn(self.proj(x))
        logits = self.head(seq)
        return {"logits": logits, "tokens": logits.argmax(dim=-1)}


def build_baseline_hw(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    input_dim: int,
    vocab_size: int,
    variant: str,
    width_mult: float = 1.0,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return CompactHandwritingRecognizer(
        family=str(family),
        input_dim=int(input_dim),
        vocab_size=int(vocab_size),
        width=width,
        depth=int(spec["depth"]),
    )


def smoke_test_hw(builder, variant: str):
    out = builder(input_dim=3, vocab_size=40, variant=variant, width_mult=0.5)(
        torch.randn(2, 32, 3)
    )
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
