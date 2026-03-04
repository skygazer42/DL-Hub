from __future__ import annotations

import torch
from torch import nn

from dlhub.nlp.utils import masked_max_pool, masked_mean_pool, sequence_lengths


class AdditiveTokenAttention(nn.Module):
    def __init__(self, dim: int, *, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        self.score = nn.Sequential(
            nn.Linear(d, d),
            nn.Tanh(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(d, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T)
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~mask.to(torch.bool), -1e9)
        w = torch.softmax(scores, dim=1)
        return (w.unsqueeze(-1) * x).sum(dim=1)


def parse_num_layers_suffix(name: str) -> tuple[str, int]:
    s = str(name).lower().strip()
    if not s:
        return s, 1

    # Accept lab-style suffixes like `mean2l`, `attn3l`, `last10l`, etc.
    # No suffix => 1 layer.
    if not s.endswith("l"):
        return s, 1

    i = len(s) - 2
    while i >= 0 and s[i].isdigit():
        i -= 1
    digits = s[i + 1 : -1]
    if not digits:
        return s, 1

    num_layers = int(digits)
    if num_layers <= 0:
        raise ValueError("Invalid num-layers suffix; expected <int>l with int > 0")
    return s[: i + 1], num_layers


def pool_sequence(
    x: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pooling: str,
    bidirectional: bool,
    attn: AdditiveTokenAttention | None,
) -> torch.Tensor:
    pooling = str(pooling).lower().strip()
    if pooling == "last":
        lengths = sequence_lengths(attention_mask).to(torch.long)
        b, t, c = x.shape
        idx = (lengths - 1).clamp(min=0, max=max(0, t - 1)).view(b, 1, 1).expand(b, 1, c)

        if bidirectional:
            if c % 2 != 0:
                raise ValueError("bidirectional sequence must have even channel dim")
            h = c // 2
            fwd = x[:, :, :h]
            bwd = x[:, :, h:]
            idx_f = (lengths - 1).clamp(min=0, max=max(0, t - 1)).view(b, 1, 1).expand(
                b, 1, h
            )
            f_last = fwd.gather(1, idx_f).squeeze(1)
            b_last = bwd[:, 0, :]
            return torch.cat([f_last, b_last], dim=-1)

        return x.gather(1, idx).squeeze(1)

    if pooling == "max":
        return masked_max_pool(x, attention_mask)
    if pooling == "mean":
        return masked_mean_pool(x, attention_mask)
    if pooling == "attn":
        if attn is None:
            raise RuntimeError("attn pooling requested but attn module missing")
        return attn(x, attention_mask)

    raise ValueError("pooling must be one of: last|max|mean|attn")


__all__ = ["AdditiveTokenAttention", "parse_num_layers_suffix", "pool_sequence"]

